import torch
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # for i, j in zip(reversed(seq), reversed(seq_next)):
        #     t = (torch.ones(n) * i).to(x.device)
        #     next_t = (torch.ones(n) * j).to(x.device)
        #     print(compute_alpha(b, t.long()).mean(),(1 - compute_alpha(b, t.long())).sqrt().mean())
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def generalized_steps_path(x, seq, model, model_path1,model_path2,timesteps, **kwargs):
    # with torch.no_grad():
    n = x.size(0)
    # seq_next = [-1] + list(seq[:-1])

    seq_next = torch.zeros_like(seq)

    seq_next[:, 0] = -1
    seq_next[:, 1:] = seq[:, :-1]
    x0_preds = []
    # xs = [x]
    seq=torch.flip(seq, dims=[1])
    seq_next=torch.flip(seq_next, dims=[1])
    # print(seq.shape,seq_next.shape)
    xs=x
    for i in range(timesteps):
        t=seq[:,i].to(x.device)
        next_t=seq_next[:,i].to(x.device)
        at = model_path1(t.unsqueeze(1).float()).view(-1, 1, 1, 1)
        bt= model_path2(t.unsqueeze(1).float()).view(-1, 1, 1, 1)
        at_next = model_path1(next_t.unsqueeze(1).float()).view(-1, 1, 1, 1)
        bt_next = model_path2(next_t.unsqueeze(1).float()).view(-1, 1, 1, 1)

        et = model(xs, t)
        et=et.detach()
        # print(xt.shape,et.shape,bt.shape)
        x0_t = (xs - et * bt) / at
        xs = at_next * x0_t + bt_next * et
    return xs

def generalized_steps_path_test(x, seq, model, model_path1,model_path2, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # xs=x
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            print(model_path1(t.unsqueeze(1).float()).mean(),model_path2(t.unsqueeze(1).float()).mean())

        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = model_path1(t.unsqueeze(1).float()).view(-1, 1, 1, 1)
            bt= model_path2(t.unsqueeze(1).float()).view(-1, 1, 1, 1)
            at_next = model_path1(next_t.unsqueeze(1).float()).view(-1, 1, 1, 1)
            bt_next = model_path2(next_t.unsqueeze(1).float()).view(-1, 1, 1, 1)

            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * bt) / at
            x0_preds.append(x0_t)
            # c1 = (
            #     kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            # )
            # c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # xs = at_next * x0_t + bt_next * et
            # xs=(xt_next)
            xt_next = at_next * x0_t + bt_next * et
            xs.append(xt_next.to('cpu'))

    # gpu_tracker.track()
    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
