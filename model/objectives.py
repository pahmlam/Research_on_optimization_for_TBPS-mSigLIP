import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_constrative(
    image_features,
    text_features,
    sim_targets,
    logit_scale,
    logit_bias,
    use_sigmoid,
):
    """
    Compute contrastive loss for image-text pairs.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    logit_t2i = logit_scale * text_features @ image_features.t() + logit_bias
    logit_i2t = logit_scale * image_features @ text_features.t() + logit_bias

    if use_sigmoid:
        loglik = F.logsigmoid(logit_t2i * sim_targets)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
    else:
        loss_i2t = -torch.sum(F.log_softmax(logit_i2t, dim=1) * sim_targets, dim=1)
        loss_t2i = -torch.sum(F.log_softmax(logit_t2i, dim=1) * sim_targets, dim=1)
        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2

    return loss


def compute_simclr(
    image_features_1,
    image_features_2,
    temperature=0.07,
):
    """
    Contrastive learning loss using SimCLR.
    """
    device = image_features_1.device
    batch_size = image_features_1.shape[0]

    image_features_1 = F.normalize(image_features_1, dim=-1, p=2)
    image_features_2 = F.normalize(image_features_2, dim=-1, p=2)

    labels = torch.arange(start=0, end=batch_size, device=device)

    sim_ab = (image_features_1 @ image_features_2.t()) / temperature
    sim_ba = sim_ab.t()

    mask = torch.where(F.one_hot(labels, batch_size) == 0, 0, float("-inf"))
    sim_aa = (image_features_1 @ image_features_1.t()) / temperature + mask
    sim_bb = (image_features_2 @ image_features_2.t()) / temperature + mask

    sim_a = torch.cat((sim_ab, sim_aa), dim=1)
    sim_b = torch.cat((sim_ba, sim_bb), dim=1)

    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    return (loss_a + loss_b) / 2


def compute_citc(
    image_features,
    text_features,
    logit_scale,
    logit_bias,
    inmodal_weight,
    intermodal_weight,
):
    """
    Compute cyclic image-text contrastive loss.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_i2i = logit_scale * image_features @ image_features.t() + logit_bias
    sim_t2t = logit_scale * text_features @ text_features.t() + logit_bias

    sim_i2t = logit_scale * image_features @ text_features.t() + logit_bias
    sim_t2i = sim_i2t.t()

    inmodal_cyclic_loss = (sim_i2i - sim_t2t).square().mean() / (
        logit_scale * logit_scale
    )
    intermodal_cyclic_loss = (sim_i2t - sim_t2i).square().mean() / (
        logit_scale * logit_scale
    )

    return (
        inmodal_weight * inmodal_cyclic_loss
        + intermodal_weight * intermodal_cyclic_loss
    )


def compute_cross_modal_circle(image_features, text_features, pids, m=0.25, gamma=128):
    """
    Circle Loss between 2 modalities.
    """
    image_features = F.normalize(image_features, dim=1, p=2)
    text_features = F.normalize(text_features, dim=1, p=2)

    sim_mat = torch.matmul(image_features, text_features.t())

    pids = pids.view(-1, 1)

    pos_mask = torch.eq(pids, pids.t()).float()
    neg_mask = 1 - pos_mask

    s_p = sim_mat[pos_mask.bool()]
    s_n = sim_mat[neg_mask.bool()]

    if s_p.numel() == 0 or s_n.numel() == 0:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + m, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + m, min=0.)

    delta_p = 1 - m
    delta_n = m

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)

    soft_plus = nn.Softplus()
    loss = soft_plus(
        torch.logsumexp(logit_p, dim=0) +
        torch.logsumexp(logit_n, dim=0)
    )

    return loss
