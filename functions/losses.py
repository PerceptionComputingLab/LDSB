import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def path_estimation_loss(x0: torch.Tensor,
                          x_theta: torch.Tensor,
                          keepdim=False):
    if keepdim:
        return (x0 - x_theta).square().sum(dim=(1, 2, 3))
    else:
        return (x0 - x_theta).square().sum(dim=(1, 2, 3)).mean(dim=0)

def denoise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          model_path1_detached,model_path2_detached, keepdim=False):
    n=x0.size(0)
    t = (torch.ones(n).to(x0.device) * t).to(x0.device)
    at = model_path1_detached(t.unsqueeze(1).float()).view(-1, 1, 1, 1).detach()
    bt = model_path2_detached(t.unsqueeze(1).float()).view(-1, 1, 1, 1).detach()
    x = x0 * at + e * bt
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'denoise': denoise_estimation_loss,
    'path': path_estimation_loss,
}
