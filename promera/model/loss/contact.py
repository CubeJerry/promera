import torch
import torch.nn.functional as F

inf = 1e9


def contact_module_loss(
    contact_logits, pred_dist, feats, multiplicity, contact_threshold=8.0
):
    """BCE loss on token pairs in contact according to the sampled structure.

    Supervises each predicted contact as true (in ground truth) or false.
    """

    gt_contacts = feats["token_contacts"].float().repeat_interleave(multiplicity, 0)

    # Pairs to supervise: both tokens present, different chains, in contact in pred
    pred_contact_mask = (pred_dist < contact_threshold).float()
    pos_mask = feats["token_pos_mask"].repeat_interleave(multiplicity, 0)
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    interchain = (asym_id[:, :, None] != asym_id[:, None, :]).float()
    mask = pos_mask[:, :, None] * pos_mask[:, None, :] * interchain

    mask = mask * pred_contact_mask

    loss = F.binary_cross_entropy_with_logits(
        contact_logits, gt_contacts, reduction="none"
    )
    loss = (loss * mask).sum((-1, -2)) / (1e-5 + mask.sum((-1, -2)))

    frac_contacts = (mask * gt_contacts).sum() / (1e-5 + mask.sum())
    return loss.mean(), frac_contacts
