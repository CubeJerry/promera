from pathlib import Path


def test_design_entrypoint_forces_single_item_loader_batches():
    source = (Path(__file__).resolve().parents[1] / "promera" / "__main__.py").read_text()

    assert 'requested_batch_size = int(task_cfg.get("batch_size", 1))' in source
    assert 'if args.task.endswith(".Design"):' in source
    assert "batch_size = 1" in source
    assert "PROMERA_DESIGN_BATCH_SIZE_OVERRIDDEN" in source
    assert "batch_size=batch_size" in source


def test_non_design_tasks_keep_their_requested_batch_size():
    source = (Path(__file__).resolve().parents[1] / "promera" / "__main__.py").read_text()

    assert "else:\n    batch_size = requested_batch_size" in source
