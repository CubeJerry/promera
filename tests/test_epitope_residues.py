import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_stubs(monkeypatch):
    dist = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "tinyprot.feature",
        types.SimpleNamespace(AF3Featurizer=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "tinyprot.msa",
        types.SimpleNamespace(
            construct_paired_msa=lambda *a, **k: None,
            load_msa_from_dir=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "tinyprot", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "promera.data.utils",
        types.SimpleNamespace(collate=lambda x: x),
    )
    utils = types.ModuleType("promera.inference.utils")
    for name in [
        "_copy_sample_to_struct",
        "_resolve_residue_idx",
        "_struct_to_pdb",
        "compute_agg_confidence",
        "compute_contact_stats",
        "compute_dockq",
        "compute_interface_contacts",
        "compute_self_consistency_rmsd",
        "compute_target_lddt",
        "finalize_feats",
        "msa_summary",
        "run_lmpnn_redesign",
    ]:
        setattr(utils, name, lambda *a, **k: None)
    utils._AA3TO1 = {"ALA": "A"}
    utils.extract_schema_target_sequence = (
        lambda schema, chain: schema[chain]["sequence"]
    )
    utils.extract_template_chain_sequence_and_ca = lambda chain: (chain.seq, [], [])
    utils.align_schema_to_template = lambda schema_seq, template_seq: (
        {i: template_seq.index(schema_seq) + i for i in range(len(schema_seq))},
        {
            "identity": 1.0,
            "coverage": 1.0,
            "mapped": len(schema_seq),
            "matches": len(schema_seq),
        },
    )
    utils.validate_alignment_or_raise = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "promera.inference.utils", utils)

    class Structure:
        @staticmethod
        def from_mmcif(path):
            return None

        @staticmethod
        def from_schema(schema):
            return None

    monkeypatch.setitem(
        sys.modules,
        "tinyprot.structure",
        types.SimpleNamespace(Structure=Structure),
    )


def _design(monkeypatch):
    _install_stubs(monkeypatch)
    sys.modules.pop("promera.inference.design", None)
    pkg = types.ModuleType("promera.inference")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "promera" / "inference")]
    monkeypatch.setitem(sys.modules, "promera.inference", pkg)
    return importlib.import_module("promera.inference.design")


class _Ridx(list):
    def tolist(self):
        return list(self)


class _Chain:
    def __init__(self, seq, ridx):
        self.seq = seq
        self.ridx = _Ridx(ridx)


class _Struct:
    def __init__(self, chain):
        self.chains = {"A": chain}


def _cfg(epitope_residues, target_template=None):
    return SimpleNamespace(
        epitope_residues=epitope_residues,
        target_template=target_template,
    )


def test_cropped_template_epitopes_are_stored_and_used_for_metrics(monkeypatch):
    d = _design(monkeypatch)
    template_seq = "A" * 98 + "B" * 200 + "C" * 20
    schema = {"A": {"type": "protein", "sequence": "B" * 200}}
    monkeypatch.setattr(
        d.Structure,
        "from_mmcif",
        staticmethod(
            lambda path: _Struct(
                _Chain(template_seq, range(1, len(template_seq) + 1))
            )
        ),
    )

    requested = [247, 248] + list(range(267, 279))
    converted = d._map_epitope_residues(
        schema,
        _cfg(requested, SimpleNamespace(path="template.cif", chain="A")),
        "A",
    )

    struct = SimpleNamespace(epitope_residues=converted)

    assert struct.epitope_residues == [149, 150] + list(range(169, 181))
    assert d._epitope_positions(200, struct.epitope_residues) == [
        148,
        149,
    ] + list(range(168, 180))


def test_untargeted_epitopes_skip_conversion_and_metrics_do_not_crash(monkeypatch):
    d = _design(monkeypatch)

    converted = d._map_epitope_residues(
        {"A": {"type": "protein", "sequence": "AAAA"}},
        _cfg(None, None),
        "A",
    )

    assert converted == []
    assert d._epitope_positions(4, converted) == []


def test_no_template_targeted_epitopes_remain_schema_positions_for_metrics(monkeypatch):
    d = _design(monkeypatch)

    converted = d._map_epitope_residues(
        {"A": {"type": "protein", "sequence": "AAAAA"}},
        _cfg([2, 5], None),
        "A",
    )

    assert converted == [2, 5]
    assert d._epitope_positions(5, converted) == [1, 4]
