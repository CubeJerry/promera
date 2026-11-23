"""Select accepted guides from a native collated batch before inference."""


def accepted_batch(batch):
    keep = [i for i, record in enumerate(batch["angle_record"])
            if record.get("record_status") != "rejected_guide"]
    if not keep:
        return None
    if len(keep) == len(batch["angle_record"]):
        return batch
    # Native collate stacks arrays as tensors and stores metadata as lists.
    return {
        key: [value[i] for i in keep] if isinstance(value, (list, tuple)) else value[keep]
        for key, value in batch.items()
    }
