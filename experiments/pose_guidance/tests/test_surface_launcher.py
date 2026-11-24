"""Exercise the host launcher and the saved NOMINEE setup boundary on CPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from promera_angles.setup_input import (
    prepare_setups,
    referenced_paths,
    validate_author_selection,
)


REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "experiments/pose_guidance/launch_surface_elevation.sh"


@pytest.mark.parametrize("residues", ["A:10,12-15;B:7", "Left:38,Left:40", "A:10"])
def test_author_prompt_accepts_qualified_groups(residues: str) -> None:
    validate_author_selection(residues)


@pytest.mark.parametrize("residues", ["", "10,12-15", "A:", "A:10;7", "A:10;"])
def test_author_prompt_rejects_ambiguous_groups(residues: str) -> None:
    with pytest.raises(ValueError, match="AUTHOR"):
        validate_author_selection(residues)


@pytest.fixture
def host_standins(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Stand in for Apptainer/Slurm; execute the actual host and container scripts."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    image = tmp_path / "promera.sif"
    image.touch()
    assets = tmp_path / "assets"
    (assets / "checkpoints").mkdir(parents=True)
    (assets / "checkpoints/promera_2606.ckpt").touch()
    setup = tmp_path / "antigen setup.yaml"
    setup.write_text("kind: nominee_antigen_setup\n")
    log = tmp_path / "calls.jsonl"
    apptainer = binaries / "apptainer"
    apptainer.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json, os, pathlib, sys
            args = sys.argv[1:]
            with open(os.environ['LAUNCH_TEST_LOG'], 'a') as handle:
                handle.write(json.dumps({'kind': 'container', 'args': args}) + '\\n')
            start = args.index('-m')
            module, payload = args[start + 1], args[start + 2:]
            if module == 'promera_angles.setup_input':
                if '--list-inputs' not in payload:
                    root = pathlib.Path(payload[3])
                    (root / 'inputs').mkdir()
                    (root / 'inputs/targeted.json').write_text('{}')
            elif '--layout-only' in payload:
                print(3 * int(payload[payload.index('--replicates') + 1]))
            """
        )
    )
    apptainer.chmod(0o755)
    scheduler = binaries / "sbatch"
    scheduler.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json, os, pathlib, sys
            path = pathlib.Path(os.environ['LAUNCH_TEST_LOG'])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            job = 1001 + sum(row['kind'] == 'submit' for row in records)
            with path.open('a') as handle:
                handle.write(json.dumps({'kind': 'submit', 'args': sys.argv[1:]}) + '\\n')
            print(str(job) + ';milton')
            """
        )
    )
    scheduler.chmod(0o755)
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    env.pop("POSE_EXTRA_BINDS", None)
    env.update(
        PATH=f"{binaries}:{env['PATH']}",
        PROMERA_IMAGE=str(image),
        PROMERA_ASSETS=str(assets),
        LAUNCH_TEST_LOG=str(log),
    )
    return env, setup, log


def invoke(
    env: dict[str, str], setup: Path, output: Path, *options: str, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), str(setup), "--output", str(output), *options],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_real_launcher_prompt_and_three_arm_scheduler(
    tmp_path: Path, host_standins: tuple[dict[str, str], Path, Path]
) -> None:
    env, setup, log = host_standins
    out = tmp_path / "study with spaces"
    result = invoke(
        env, setup, out,
        "--elevation", "70", "90", "--legacy-tilt", "0", "20",
        "--backbones", "7", "--replicates", "3", "--seed", "13",
        stdin="A:10,12-15;B:7\n",
    )
    assert result.returncode == 0, result.stderr
    assert "Target AUTHOR residues" in result.stderr
    assert f"Report: {out}/analysis/report.html" in result.stdout
    records = [json.loads(line) for line in log.read_text().splitlines()]
    preparation = next(
        row["args"] for row in records
        if row["kind"] == "container" and "promera_angles.setup_input" in row["args"]
        and "--list-inputs" not in row["args"]
    )
    assert "A:10,12-15;B:7" in preparation
    assert "--surface-study" in preparation
    assert any("/opt/angle-defaults.yaml:ro" in arg for arg in preparation)
    submitted = [row["args"] for row in records if row["kind"] == "submit"]
    assert len(submitted) == 3
    assert "--three-arms" in submitted[0]
    assert "--elevation" in submitted[0] and "--legacy-tilt" in submitted[0]
    assert "--array=0-8%4" in submitted[1]
    assert "--dependency=afterok:1001" in submitted[1]
    assert "--dependency=afterany:1002" in submitted[2]
    assert "--ntasks=1" in submitted[0]


def test_preview_is_noninteractive_and_never_submits(
    tmp_path: Path, host_standins: tuple[dict[str, str], Path, Path]
) -> None:
    env, setup, log = host_standins
    result = invoke(env, setup, tmp_path / "preview", "--residues", "A:10", "--preview")
    assert result.returncode == 0, result.stderr
    assert "Target AUTHOR residues" not in result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(row["kind"] != "submit" for row in records)
    assert "--preview" in records[-1]["args"]
    assert "--nv" not in records[-1]["args"]


def test_existing_allocation_runs_directly(
    tmp_path: Path, host_standins: tuple[dict[str, str], Path, Path]
) -> None:
    env, setup, log = host_standins
    env["SLURM_JOB_ID"] = "123"
    result = invoke(env, setup, tmp_path / "direct", "--residues", "A:10")
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(row["kind"] != "submit" for row in records)
    assert "--nv" in records[-1]["args"]
    assert "--three-arms" in records[-1]["args"]


def test_legacy_angle_is_rejected_without_submission(
    tmp_path: Path, host_standins: tuple[dict[str, str], Path, Path]
) -> None:
    env, setup, log = host_standins
    result = invoke(env, setup, tmp_path / "invalid", "--angle", "0", "20")
    assert result.returncode == 2
    assert "retains legacy tilt semantics" in result.stderr
    assert not log.exists()


def test_noninteractive_missing_selection_fails_before_work(
    tmp_path: Path, host_standins: tuple[dict[str, str], Path, Path]
) -> None:
    env, setup, log = host_standins
    result = invoke(env, setup, tmp_path / "invalid")
    assert result.returncode == 2
    assert "--residues" in result.stderr
    assert not log.exists()


@pytest.fixture
def nominee_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional installed NOMINEE source; enable with NOMINEE_TEST_SOURCE."""
    if source := os.environ.get("NOMINEE_TEST_SOURCE"):
        monkeypatch.syspath_prepend(source)
    pytest.importorskip(
        "utils.nominee_core.promera", reason="requires installed NOMINEE source"
    )


def test_nominee_setup_uses_author_numbers_and_default_framework(
    tmp_path: Path, nominee_source: None
) -> None:
    from promera_angles.demo import write_cif

    cif = tmp_path / "antigen.cif"
    xyz = np.array([[i * 3.8, 0.0, 0.0] for i in range(6)])
    write_cif(cif, {"A": ("ASGASG", xyz), "B": ("SGASGA", xyz + [0, 9, 0])})
    # NOMINEE uses Bio.PDB's full parser; the minimal geometry writer omits B-factors.
    text = cif.read_text()
    if "_atom_site.B_iso_or_equiv" not in text:
        cif.write_text(
            "\n".join(
                line[:-1] + "20 1"
                if line.startswith("ATOM ")
                else line.replace(
                    "_atom_site.pdbx_PDB_model_num",
                    "_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num",
                )
                for line in text.splitlines()
            ) + "\n"
        )
    setup = tmp_path / "antigen_setup.json"
    setup.write_text(json.dumps({
        "kind": "nominee_antigen_setup", "schema_version": 3,
        "antigen_cif": "antigen.cif", "template_cif": "antigen.cif",
        "target_chain": "A", "target_chains": ["A", "B"], "template_chain": "A",
        "epitope_numbering": "runtime", "epitope_residues": "A:1",
        "epitope_residue_map": {"A": [1]},
    }))
    original = setup.read_bytes()
    out = tmp_path / "study"
    out.mkdir()
    cfg = prepare_setups(
        setup, None, "A:38,40;B:39", out,
        surface_study=True, defaults=REPO / "examples/design_vhh.yaml",
    )
    run = json.loads((out / "inputs/nominee_run.json").read_text())
    assert run["epitope_numbering"] == "author"
    assert run["epitope_residue_map"] == {"A": [38, 40], "B": [39]}
    assert cfg["epitope_residue_map"] == {"A": [1, 3], "B": [2]}
    assert cfg["binder"]["chain"] == "C"
    assert cfg["inverse_folder"]["num_seqs"] == 1
    assert cfg["inverse_folder"]["type"] == "solublempnn"
    assert "approach_angle" not in cfg
    assert cfg["binder"]["cdr_lengths"] == {
        "cdrh1": [6, 8], "cdrh2": [6, 8], "cdrh3": [8, 20]
    }
    assert setup.read_bytes() == original
    assert Path(cfg["target_template"]["path"]).is_file()
    provenance = json.loads((out / "inputs/setup_provenance.json").read_text())
    assert provenance["framework_default_used"]
    assert provenance["native_refolds_per_sequence"] == 5


def test_reference_override_regenerates_schema_without_stale_paths(
    tmp_path: Path, nominee_source: None
) -> None:
    setup = tmp_path / "antigen_setup.json"
    setup.write_text(json.dumps({
        "antigen_cif": "old.cif", "template_cif": "old.cif",
        "promera_input": "old_schema.json",
    }))
    replacement = tmp_path / "new.cif"
    assert referenced_paths(setup, replacement) == {
        "antigen_cif": str(replacement), "template_cif": str(replacement)
    }
    setup.write_text(json.dumps({
        "antigen_cif": "old.cif", "template_cif": "separate.cif",
        "promera_input": "old_schema.json",
    }))
    assert referenced_paths(setup, replacement)["template_cif"] == str(
        tmp_path / "separate.cif"
    )
