# Adapted from https://github.com/jwohlwend/boltz
import torch


def distogram_loss(pred, feats, breakdown=False):
    # Compute target distogram
    t_dists = torch.cdist(feats["token_pos"], feats["token_pos"])
    # target = feats["disto_target"]

    # TODO: fix the magic numbers here
    boundaries = torch.linspace(2.0, 22.0, 64 - 1).to(t_dists)
    target = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()

    # Combine target mask and padding mask
    mask = feats["token_pos_mask"]
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)
    if "distogram_supervise" in feats:
        mask = mask * feats["distogram_supervise"]

    errors = torch.nn.functional.cross_entropy(
        pred.permute(0, 3, 1, 2), target, reduction="none"
    )
    loss = (errors * mask).sum((-1, -2)) / (1e-5 + mask.sum((-1, -2)))

    if not breakdown:
        return loss

    breakdown = {}
    for i in feats["asym_id"].unique():
        for j in feats["asym_id"].unique():
            i = int(i)
            j = int(j)
            new_mask = (
                mask
                * (feats["asym_id"] == i).float().unsqueeze(-1)
                * (feats["asym_id"] == j).float().unsqueeze(-2)
            )
            this_loss = (errors * new_mask).sum((-1, -2)) / (
                1e-5 + new_mask.sum((-1, -2))
            )
            weights = new_mask.sum((-1, -2))
            breakdown[(i, j)] = this_loss.item(), weights.item()

    return loss, breakdown
