"""Fixed, single-protein GFP geometry benchmark (UniProt P42212)."""

from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

GFP_PAIR_COUNTS = {
    "ordinary": 0,
    "gfp_pairs_04": 4,
    "gfp_pairs_08": 8,
    "gfp_pairs_12": 12,
    "gfp_pairs_16": 16,
}

# Reviewed Aequorea victoria GFP, P42212. Permit close full-length GFP variants.
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def validate_gfp_sequences(sequences: dict[str, str]) -> None:
    if len(sequences) != 1:
        raise ValueError("GFP pair ablation requires one GFP antigen chain")
    seq = next(iter(sequences.values())).upper()
    matches = sum(
        block.size
        for block in SequenceMatcher(None, GFP, seq, autojunk=False).get_matching_blocks()
    )
    if not 220 <= len(seq) <= 265 or matches / max(len(seq), len(GFP)) < 0.90:
        raise ValueError("GFP pair ablation is restricted to full-length avGFP variants")


def validate_gfp_plan(plan: dict[str, Any]) -> None:
    sequences = plan["sequences"]
    validate_gfp_sequences(sequences)
    schema = json.loads(Path(plan["input"]).read_text())
    if set(schema) - {"connections"} != set(sequences):
        raise ValueError("GFP benchmark requires a GFP-only input schema")
    for chain, seq in sequences.items():
        if schema[chain].get("type") != "protein" or schema[chain]["sequence"] != seq:
            raise ValueError("GFP benchmark schema does not match the resolved sequence")


def retain_gfp_pairs(
    bins: Any, mask: Any, records: list[dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    """Nested subsets of a fully validated 16-pair guide, no new placement."""
    count = GFP_PAIR_COUNTS[arm]
    if len(records) != 16 or count == 0:
        raise ValueError("expected a complete 16-pair GFP guide and a guided arm")
    for row in records[count:]:
        a, b = row["target_token"], row["binder_token"]
        mask[a, b] = mask[b, a] = False
        bins[a, b] = bins[b, a] = 0
    return records[:count]
