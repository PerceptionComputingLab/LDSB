import torch
from .losses import loss_registry
import math
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
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            print(t.mean(),compute_alpha(b, t.long()).mean(),(1 - compute_alpha(b, t.long())).sqrt().mean())
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            print(et.min(),et.max())

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


stepnum1=1

def generalized_steps_path1b(combined_batche_size,device,combined_batches,nn,b, ts,tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b,num_timesteps, **kwargs):
    # with torch.no_grad():

    # n = x.size(0)
    # z=torch.randn_like(x)
    tl=ts
    ts = [-1/1000] + list(ts[:-1])
    n=combined_batche_size
    allt=np.append(ts,tl[-1])
    allt=tl
    # print(tl,ts)
    model.eval()
    traj = [] # to store the trajectory
    signal = []
    tllist = []
    tslist = []
    loss=[]
    a1=1
    b1=1
    for i in range(len(ts)):
        if i >= len(ts):
            t = tl[i - 1]
        else:
            t = tl[i]

        t1 = (torch.ones(n)).to(device) * t
        print(comput_a1(models2_a, t1.unsqueeze(1).float(), allt).mean(),
              comput_a2(models2_b, t1.unsqueeze(1).float(), allt).mean())
    print('----------------------------------------------')

    for jj in range(len(ts)):
        # jj=nn

        i=len(ts)-1-jj
        if ts[i] < 0:
            continue
        # for kk in range(math.ceil((jj+1)/2)):
        if jj==0:
            stepnum=stepnum1
        else:
            stepnum=1
        for kk in range(stepnum):
            z_t_batch=[]
            et_batch = []
            target_batch=[]
            for _, (x, y) in enumerate(combined_batches):
                x=x.to(device)
                dz = torch.randn_like(x)
                n=x.size(0)
                with torch.no_grad():
                    t1 = (torch.ones(n)).to(x.device)* ts[i]
                    t2 = (torch.ones(n)).to(x.device)* tl[i]
                    t11 = t1.unsqueeze(1).float()
                    t22 = t2.unsqueeze(1).float()


                    at=comput_a1(models1_a,t11,allt)
                    bt=comput_a2(models1_b,t11,allt)
                    at_next=comput_a1(models1_a,t22,allt)
                    bt_next=comput_a2(models1_b,t22,allt)
                    # print(ts[i],tl[i],at.mean(),at_next.mean(),bt.mean(),bt_next.mean())
                    at = at.mean()
                    at_next = at_next.mean()
                    bt = bt.mean()
                    bt_next = bt_next.mean()

                    target=at*x+bt*dz
                    target=target.detach()
                    z_t = at_next*x+bt_next*dz

                    z_t=z_t.detach()
                    z_t_batch.append(z_t)
                    target_batch.append(target)
                    et = model(z_t, (torch.ones(x.size(0))).to(x.device) * tl[i]*1000)
                    # et=dz
                    et_batch.append(et)

            z_t_batch = torch.cat(z_t_batch, dim=0)
            et_batch = torch.cat(et_batch, dim=0)
            target_batch = torch.cat(target_batch, dim=0)
            # z_t_batch = torch.tensor(z_t_batch)
            # et_batch = torch.tensor(et_batch)
            # target_batch = torch.tensor(target_batch)
            n=combined_batche_size
            t1 = (torch.ones(n)).to(x.device) * ts[i]
            t11 = t1.unsqueeze(1).float()
            t2 = (torch.ones(n)).to(x.device) * tl[i]
            t22 = t2.unsqueeze(1).float()

            prod = comput_a2(models2_b, t22, allt)
            prod = prod.view(-1, 1, 1, 1).detach()
            # print(t22.mean(),prod.mean())

            # bt = model2_path2(t11).view(-1, 1, 1, 1)
            btt = models2_b[f'{i-1}'](t11).view(-1, 1, 1, 1)


            B = prod*et_batch

            A = target_batch - comput_a1(models2_a, t11, allt) * (z_t_batch - bt_next.mean() * et_batch) / at_next
            # A = target_batch - comput_a1(models2_a, t11, allt) * (z_t_batch)

            A_flat = A.view(-1)
            B_flat = B.view(-1)

            epsilon = 1e-8
            # tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))+at_next1
            tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))



            optimizer2_b[f'{i-1}'].zero_grad()
            loss1=loss_registry["path"](btt, (tt).detach())

            loss1.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                        models2_b[f'{i-1}'].parameters(), 1
                    )
            except Exception:
                pass
            optimizer2_b[f'{i-1}'].step()
            scheduler2_b[f'{i-1}'].step(loss1)
            print('++++++',btt.mean(),tt.mean(),(tt-btt).mean())


            loss.append(loss1)
    return loss

def generalized_steps_path1a(combined_batche_size,device,combined_batches,nn,b, ts,tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b,num_timesteps, **kwargs):
    # with torch.no_grad():

    # n = x.size(0)
    # z=torch.randn_like(x)
    tl=ts
    ts = [-1/1000] + list(ts[:-1])
    n=combined_batche_size
    allt=np.append(ts,tl[-1])
    allt=tl
    # print(tl,ts)
    model.eval()
    traj = [] # to store the trajectory
    signal = []
    tllist = []
    tslist = []
    loss=[]
    a1=1
    b1=1
    for i in range(len(ts) ):
        if i >= len(ts):
            t = tl[i - 1]
        else:
            t = tl[i]

        t1 = (torch.ones(n)).to(device) * t
    #     print(comput_a1(models2_a, t1.unsqueeze(1).float(), allt).mean(),
    #           comput_a2(models2_b, t1.unsqueeze(1).float(), allt).mean())
    # print('----------------------------------------------')

    for jj in range(len(ts)):
        # jj=nn

        i=len(ts)-1-jj
        # print(i,jj)
        if ts[i] < 0:
            continue
        # for kk in range(math.ceil((jj+1)/2)):
        if jj == 0:
            stepnum = stepnum1
        else:
            stepnum = 1
        for kk in range(stepnum):
            z_t_batch=[]
            et_batch = []
            target_batch=[]
            for _, (x, y) in enumerate(combined_batches):
                x=x.to(device)
                dz = torch.randn_like(x)
                n=x.size(0)
                with torch.no_grad():
                    t1 = (torch.ones(n)).to(x.device)* ts[i]
                    t2 = (torch.ones(n)).to(x.device)* tl[i]
                    t11 = t1.unsqueeze(1).float()
                    t22 = t2.unsqueeze(1).float()

                    at=comput_a1(models1_a,t11,allt)
                    bt=comput_a2(models1_b,t11,allt)
                    at_next=comput_a1(models1_a,t22,allt)
                    bt_next=comput_a2(models1_b,t22,allt)
                    # print(ts[i],tl[i],at.mean(),at_next.mean(),bt.mean(),bt_next.mean())
                    at = at.mean()
                    at_next = at_next.mean()
                    bt = bt.mean()
                    bt_next = bt_next.mean()

                    target=at*x+bt*dz
                    target=target.detach()
                    z_t = at_next*x+bt_next*dz

                    z_t=z_t.detach()
                    z_t_batch.append(z_t)
                    target_batch.append(target)
                    et = model(z_t, (torch.ones(x.size(0))).to(x.device) * tl[i]*1000)
                    # et=dz
                    et_batch.append(et)

            z_t_batch = torch.cat(z_t_batch, dim=0)
            et_batch = torch.cat(et_batch, dim=0)
            target_batch = torch.cat(target_batch, dim=0)
            # z_t_batch = torch.tensor(z_t_batch)
            # et_batch = torch.tensor(et_batch)
            # target_batch = torch.tensor(target_batch)
            n=combined_batche_size
            t1 = (torch.ones(n)).to(x.device) * ts[i]
            t11 = t1.unsqueeze(1).float()
            t2 = (torch.ones(n)).to(x.device) * ts[i-1]
            t22 = t2.unsqueeze(1).float()



            att = models2_a[f'{i-1}'](t11).view(-1, 1, 1, 1)
            B = (z_t_batch - bt_next * et_batch) * comput_a1(models2_a, t22, allt) / at_next
            # B = (z_t_batch ) * comput_a1(models2_a, t22, allt)
            A = target_batch - comput_a2(models2_b, t11, allt) * et_batch
            A_flat = A.view(-1)
            B_flat = B.view(-1)
            epsilon = 1e-8
            # tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))+at_next1
            tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))


            optimizer2_a[f'{i-1}'].zero_grad()
            loss1=loss_registry["path"](att, (tt).detach())

            loss1.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                        models2_a[f'{i-1}'].parameters(), 1
                    )
            except Exception:
                pass
            optimizer2_a[f'{i-1}'].step()
            scheduler2_a[f'{i-1}'].step(loss1)
            # print('++++++',model_path2(t11).view(-1, 1, 1, 1).mean(),tt.mean(),(tt-bt).mean(),tslist[i].mean())


            loss.append(loss1)
    return loss


# 优化1b
def generalized_steps_path2b(combined_batche_size,device,combined_batches,nn,b, ts,tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b,num_timesteps, **kwargs):
    # with torch.no_grad():

    # n = x.size(0)
    # z=torch.randn_like(x)
    tl=ts
    ts = [-1/1000] + list(ts[:-1])
    n=combined_batche_size
    allt=np.append(ts,tl[-1])
    allt=tl
    # print(tl,ts)
    model.eval()
    traj = [] # to store the trajectory
    signal = []
    tllist = []
    tslist = []
    loss=[]
    a1=1
    b1=1
    for i in range(len(ts) ):
        if i >= len(ts):
            t = tl[i - 1]
        else:
            t = tl[i]

        t1 = (torch.ones(n)).to(device) * t
        print(comput_a1(models1_a, t1.unsqueeze(1).float(), allt).mean(),
              comput_a2(models1_b, t1.unsqueeze(1).float(), allt).mean())
    print('----------------------------------------------')

    for jj in range(len(ts)):
        # jj=nn

        i=len(ts)-1-jj
        # print(i,jj)
        if ts[i] < 0:
            continue
        # for kk in range(math.ceil((jj+1)/2)):
        if jj == 0:
            stepnum = stepnum1
        else:
            stepnum = 1
        for kk in range(stepnum):
            z_t_batch=[]
            et_batch = []
            target_batch=[]
            for _, (x, y) in enumerate(combined_batches):
                x=x.to(device)
                dz = torch.randn_like(x)
                n=x.size(0)
                with torch.no_grad():
                    t1 = (torch.ones(n)).to(x.device)* ts[i]
                    t2 = (torch.ones(n)).to(x.device)* tl[i]
                    t11 = t1.unsqueeze(1).float()
                    t22 = t2.unsqueeze(1).float()

                    at=comput_a1(models2_a,t11,allt)
                    bt=comput_a2(models2_b,t11,allt)
                    at_next=comput_a1(models2_a,t22,allt)
                    bt_next=comput_a2(models2_b,t22,allt)
                    # print(ts[i],tl[i],at.mean(),at_next.mean(),bt.mean(),bt_next.mean())
                    at = at.mean()
                    at_next = at_next.mean()
                    bt = bt.mean()
                    bt_next = bt_next.mean()

                    target = at_next * x + bt_next * dz
                    et = model(target, t2 * 1000)
                    # et=dz

                    x0_t = (target - et * bt_next) / at_next
                    z_t = at * x0_t + bt * et


                    # z_t_batch.append(x0_t)
                    z_t_batch.append(z_t)
                    target_batch.append(target)
                    et_batch.append(dz)

            z_t_batch = torch.cat(z_t_batch, dim=0)
            et_batch = torch.cat(et_batch, dim=0)
            target_batch = torch.cat(target_batch, dim=0)

            n=combined_batche_size
            if i+1>=len(tl):

                continue
            else:
                t1 = (torch.ones(n)).to(x.device) * tl[i+1]
                t11 = t1.unsqueeze(1).float()
                prod = comput_a2(models1_b, t11, allt)
                prod = prod.view(-1, 1, 1, 1)
                prod=prod.mean()
            # print(prod.mean())
            t2 = (torch.ones(n)).to(x.device) * tl[i]
            t22 = t2.unsqueeze(1).float()


            A=target_batch - comput_a1(models1_a,t22,allt) * (z_t_batch - bt.mean() * et_batch)/at
            # A=target_batch - comput_a1(models1_a,t22,allt) * (z_t_batch )

            B = et_batch*prod

            A_flat = A.view(-1)
            B_flat = B.view(-1)
            epsilon = 1e-8
            # tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))+at_next1
            tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))
            btt = models1_b[f'{i}'](t22).view(-1, 1, 1, 1)
            # print(t22.mean(),comput_a2(models1_b, t22, allt).mean())

            # print('------',bt.mean(),tt.mean(),(tt-bt).mean())
            optimizer1_b[f'{i}'].zero_grad()
            loss1=loss_registry["path"](btt, (tt).detach())

            loss1.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                        models1_b[f'{i}'].parameters(), 1
                    )
            except Exception:
                pass
            optimizer1_b[f'{i}'].step()
            scheduler1_b[f'{i}'].step(loss1)


            loss.append(loss1)
    return loss


def generalized_steps_path2a(combined_batche_size,device,combined_batches,nn,b, ts,tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b,num_timesteps, **kwargs):
    # with torch.no_grad():

    # n = x.size(0)
    # z=torch.randn_like(x)
    tl=ts
    ts = [-1/1000] + list(ts[:-1])
    n=combined_batche_size
    allt=np.append(ts,tl[-1])
    allt=tl
    # print(tl,ts)
    model.eval()
    traj = [] # to store the trajectory
    signal = []
    tllist = []
    tslist = []
    loss=[]
    a1=1
    b1=1
    for i in range(len(ts) ):
        if i >= len(ts):
            t = tl[i - 1]
        else:
            t = tl[i]

        t1 = (torch.ones(n)).to(device) * t
    #     print(comput_a1(models1_a, t1.unsqueeze(1).float(), allt).mean(),
    #           comput_a2(models1_b, t1.unsqueeze(1).float(), allt).mean())
    # print('----------------------------------------------')
    for jj in range(len(ts)):
        # jj=nn

        i = len(ts) - 1 - jj
        # print(i,jj)
        if ts[i] < 0:
            continue
        # for kk in range(math.ceil((jj + 1) / 2)):
        if jj == 0:
            stepnum = stepnum1
        else:
            stepnum = 1
        for kk in range(stepnum):
            z_t_batch = []
            et_batch = []
            target_batch = []
            for _, (x, y) in enumerate(combined_batches):
                x = x.to(device)
                dz = torch.randn_like(x)
                n = x.size(0)
                with torch.no_grad():
                    t1 = (torch.ones(n)).to(x.device) * ts[i]
                    t2 = (torch.ones(n)).to(x.device) * tl[i]
                    t11 = t1.unsqueeze(1).float()
                    t22 = t2.unsqueeze(1).float()

                    at = comput_a1(models2_a, t11, allt)
                    bt = comput_a2(models2_b, t11, allt)
                    at_next = comput_a1(models2_a, t22, allt)
                    bt_next = comput_a2(models2_b, t22, allt)
                    # print(ts[i],tl[i],at.mean(),at_next.mean(),bt.mean(),bt_next.mean())
                    at = at.mean()
                    at_next = at_next.mean()
                    bt = bt.mean()
                    bt_next = bt_next.mean()

                    target = at_next * x + bt_next * dz
                    et = model(target, t2 * 1000)
                    # et=dz

                    x0_t = (target - et * bt_next) / at_next
                    z_t = at * x0_t + bt * et

                    # z_t_batch.append(x0_t)
                    z_t_batch.append(z_t)
                    target_batch.append(target)
                    et_batch.append(dz)

            z_t_batch = torch.cat(z_t_batch, dim=0)
            et_batch = torch.cat(et_batch, dim=0)
            target_batch = torch.cat(target_batch, dim=0)

            n=combined_batche_size
            t1 = (torch.ones(n)).to(x.device) * ts[i]
            t11 = t1.unsqueeze(1).float()
            t2 = (torch.ones(n)).to(x.device) * tl[i]
            t22 = t2.unsqueeze(1).float()

            if i+1>=len(tl):
                continue
            A = target_batch - comput_a2(models1_b, t22, allt) * et_batch
            B = (z_t_batch - bt * et_batch) * comput_a1(models1_a, t11, allt) / at
            # B = (z_t_batch ) * comput_a1(models1_a, t11, allt)
            A_flat = A.view(-1)
            B_flat = B.view(-1)
            epsilon = 1e-8
            # tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))+at_next1
            tt = (torch.dot(A_flat, B_flat) / ((torch.dot(B_flat, B_flat))))

            att = models1_a[f'{i}'](t22).view(-1, 1, 1, 1)
            # print(t22.mean(),comput_a1(models1_a, t22, allt).mean())

            optimizer1_a[f'{i}'].zero_grad()
            loss1 = loss_registry["path"](att, (tt).detach())

            loss1.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    models1_a[f'{i}'].parameters(), 1
                )
            except Exception:
                pass
            optimizer1_a[f'{i}'].step()
            scheduler1_a[f'{i}'].step(loss1)

            loss.append(loss1)
    return loss



def comput_a1(models2_a,t,seq):
    t=t.unsqueeze(1).float()
    a1=torch.ones_like(t)
    for i in range(seq.shape[0]):
        if seq[i] <= t.mean():
            if seq[i].mean() == 1:
                a = torch.ones_like(a1)*0.02
            elif seq[i].mean() <= 0:
                a = torch.ones_like(a1)
            else:
                a=models2_a[f'{i}'](seq[i]*torch.ones_like(t))
            a1=a1*a
    return a1.view(-1, 1, 1, 1)

def comput_a2(models2_b,t,seq):
    t=t.unsqueeze(1).float()
    a1=torch.ones_like(t)
    for i in range(seq.shape[0]):
        if seq[i] >= t.mean():
            if seq[i].mean() == 1:
                a = torch.ones_like(a1)*1
            elif seq[i].mean() < 0:
                a = torch.ones_like(a1)*0.02
            else:
                a = models2_b[f'{i}'](seq[i]*torch.ones_like(t))
            a1 = a1 * a
    return a1.view(-1, 1, 1, 1)

import numpy as np
def generalized_steps_path_test(aa,bb,x, seq, model, models2_a,models2_b,b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for index,(i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        # for index, (i, j) in enumerate(zip((seq), (seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            bt = (1 - at).sqrt()
            bt_next = (1-at_next).sqrt()
            at=at.sqrt()
            at_next=at_next.sqrt()
            xt = xs[-1].to('cuda')
            et = model(xt, t)


            at=aa[index]
            bt=bb[index]
            if index+1<len(aa):
                at_next=aa[index+1]
                bt_next=bb[index+1]
            else:
                at_next = comput_a1(models2_a, next_t / 1000, np.array(seq) / 1000)
                bt_next = comput_a2(models2_b, next_t / 1000, np.array(seq) / 1000)
                # at_next = at
                # bt_next = bt

            # print('+++',t.mean(),next_t.mean(),at.mean(),bt.mean())
            # print('+++',t.mean(),next_t.mean(),at_next.mean(),bt_next.mean())


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



    # 原始的
    # return xs, x0_preds
    return  x0_preds,xs

def print_testpath(x, seq, model, models2_a,models2_b,b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # print(seq,seq_next)
        aa=[]
        bb=[]
        tt=[]
        an=[]
        for i, j in zip(reversed(seq), reversed(seq_next)):
        # for i, j in zip((seq), (seq_next)):
            # print(i)
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            # at_next = compute_alpha(b, next_t.long())
            bt = (1 - at).sqrt()
            at=at.sqrt()
            # xt = xs[-1].to('cuda')
            # et = model(xt, t)
            at=comput_a1(models2_a,t/1000,np.array(seq) / 1000)
            bt = comput_a2(models2_b, t / 1000, np.array(seq) / 1000)

            tt.append(t.mean())
            aa.append(at.mean())
            bb.append(bt.mean())
            print(at.mean(),bt.mean())



    return aa,bb



