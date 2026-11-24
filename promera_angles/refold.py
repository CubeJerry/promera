"""Guard the native fresh-feature refolding path against generation restraints."""

from __future__ import annotations

from typing import Any


class UnconditionedRefoldMixin:
    """Refolds must reach the trunk without guide distances or epitope flags."""

    def _refold_batch(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        original = model.pairformer_forward

        def unrestrained(feats: dict[str, Any], *a: Any, **kw: Any) -> Any:
            for key in ("distogram_mask", "is_epitope"):
                value = feats.get(key)
                if value is not None and bool(value.any()):
                    raise RuntimeError(
                        f"validation contamination: nonzero {key} in refold"
                    )
            return original(feats, *a, **kw)

        model.pairformer_forward = unrestrained
        try:
            refold = getattr(super(), "_refold_batch")
            return refold(model, *args, **kwargs)
        finally:
            model.pairformer_forward = original
