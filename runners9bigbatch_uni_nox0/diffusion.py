import os
import logging
import time
import glob
import random
import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model,MLP_path,KAN_path
from models.kan import KAN
from models.ema import EMAHelper
from .get_optimizer import get_optimizer
from .losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from .ckpt_util import get_ckpt_path

import torchvision.utils as tvu

import matplotlib.pyplot as plt

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self.stru = [1, 4,16,4,1]
        self.grid_num=2
        # 0 is KAN，1 is MLP
        self.model_type=0




    def train_path(self):
        seed=0
        # 设置全局随机数种子
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        num=len(dataset)
        print(dataset)
        print(test_dataset)
        from torch.utils.data import Subset
        pre_epoch=15

        combined_batche_size=128
        # combined_batche_size=16
        step_num=100
        savefreq=step_num//5

        FB=['1','2']
        subset_indices = list(range(step_num*combined_batche_size))
        dataset1 = Subset(dataset, subset_indices)
        train_loader = data.DataLoader(
            dataset1,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        subset_indices = list(range(num-15000,num))
        # test_dataset = Subset(dataset, subset_indices)
        test_loader = data.DataLoader(
            # test_dataset,
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        # import torchvision.utils as vutils
        # save_dir = '/Share8/qiuxingyu/CIFAR10/'
        # save_dir = '/Share8/qiuxingyu/CelebA/'
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 初始化计数器
        # count = 0
        # max_images = 10000
        #
        # # 遍历 train_loader 中的每一个 batch
        # for i, (images, labels) in enumerate(test_loader):
        #     # 遍历每一个 batch 中的每一个图片
        #     for j, image in enumerate(images):
        #         # if count >= max_images:
        #         #     break
        #         # 保存彩色图像
        #         vutils.save_image(image, os.path.join(save_dir, f'image_{count}.png'))
        #         count += 1
        #     # if count >= max_images:
        #     #     break
        # test_loader = data.DataLoader(
        #     test_dataset,
        #     # dataset,
        #     batch_size=config.training.batch_size,
        #     shuffle=True,
        #     num_workers=config.data.num_workers,
        # )
        # for i, (images, labels) in enumerate(test_loader):
        #     # 遍历每一个 batch 中的每一个图片
        #     for j, image in enumerate(images):
        #         # if count >= max_images:
        #         #     break
        #         # 保存彩色图像
        #         vutils.save_image(image, os.path.join(save_dir, f'image_{count}.png'))
        #         count += 1
        #     # if count >= max_images:
        #     #     break
        #
        # print("Images saved successfully!")
        # a+1

        start_epoch, step = 0, 0

        # 5步
        # patience = int(2*len(dataset) / self.config.training.batch_size)
        # 10步
        # patience = int(5*len(dataset) / self.config.training.batch_size)
        # # 20步
        patience = int(20000*len(dataset) / self.config.training.batch_size)

        model_type=self.model_type

        models1_a = {}
        optimizer1_a = {}
        scheduler1_a = {}

        if model_type==0:
            model_path=KAN(self.stru,grid_size=self.grid_num*self.args.timesteps,grid_range=[0,1]).to(self.device)
        elif model_type==1:
            model_path = MLP_path(config).to(self.device)
            model_path.initialize()

        optimizer_path=get_optimizer(self.config, model_path.parameters())
        scheduler_path=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_path, mode='min',
                                                                              factor=0.5,
                                                                              patience=patience,
                                                                              verbose=True,
                                                                              threshold=1e-12, threshold_mode='rel',
                                                                              cooldown=int(patience / 10), min_lr=1e-11,
                                                                              eps=1e-18)
        for i in range(self.args.timesteps):
            models1_a[f'{i}'] = model_path
            optimizer1_a[f'{i}'] = optimizer_path
            scheduler1_a[f'{i}'] = scheduler_path

        models1_b = {}
        optimizer1_b = {}
        scheduler1_b = {}

        if model_type==0:
            model_path=KAN(self.stru,grid_size=self.grid_num*self.args.timesteps,grid_range=[0,1]).to(self.device)
        elif model_type==1:
            model_path = MLP_path(config).to(self.device)
            model_path.initialize()
        optimizer_path = get_optimizer(self.config, model_path.parameters())
        scheduler_path = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_path, mode='min',
                                                                    factor=0.5,
                                                                    patience=patience,
                                                                    verbose=True,
                                                                    threshold=1e-12, threshold_mode='rel',
                                                                    cooldown=int(patience / 10), min_lr=1e-11,
                                                                    eps=1e-18)
        for i in range(self.args.timesteps):
            models1_b[f'{i}'] = model_path
            optimizer1_b[f'{i}'] = optimizer_path
            scheduler1_b[f'{i}'] = scheduler_path

        models2_a = {}
        optimizer2_a = {}
        scheduler2_a = {}
        if model_type==0:
            model_path=KAN(self.stru,grid_size=self.grid_num*self.args.timesteps,grid_range=[0,1]).to(self.device)
        elif model_type==1:
            model_path = MLP_path(config).to(self.device)
            model_path.initialize()
        optimizer_path = get_optimizer(self.config, model_path.parameters())
        scheduler_path = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_path, mode='min',
                                                                    factor=0.5,
                                                                    patience=patience,
                                                                    verbose=True,
                                                                    threshold=1e-12, threshold_mode='rel',
                                                                    cooldown=int(patience / 10), min_lr=1e-11,
                                                                    eps=1e-18)
        for i in range(self.args.timesteps):
            models2_a[f'{i}'] = model_path
            optimizer2_a[f'{i}'] = optimizer_path
            scheduler2_a[f'{i}'] = scheduler_path

        models2_b = {}
        optimizer2_b = {}
        scheduler2_b = {}
        if model_type==0:
            model_path=KAN(self.stru,grid_size=self.grid_num*self.args.timesteps,grid_range=[0,1]).to(self.device)
        elif model_type==1:
            model_path = MLP_path(config).to(self.device)
            model_path.initialize()
        optimizer_path = get_optimizer(self.config, model_path.parameters())
        scheduler_path = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_path, mode='min',
                                                                    factor=0.5,
                                                                    patience=patience,
                                                                    verbose=True,
                                                                    threshold=1e-12, threshold_mode='rel',
                                                                    cooldown=int(patience / 10), min_lr=1e-11,
                                                                    eps=1e-18)
        import  math
        for i in range(self.args.timesteps):
            models2_b[f'{i}'] = model_path
            optimizer2_b[f'{i}'] = optimizer_path
            scheduler2_b[f'{i}'] = scheduler_path
        # model_path1 = torch.nn.DataParallel(model_path1)
        # model_path2 = torch.nn.DataParallel(model_path2)

        # model_path1.load_state_dict(model_path2.state_dict())
        if self.args.resume_training:

            ss='_pre'
            for i in range(1):
                states = torch.load(os.path.join(self.args.log_path, f'path_a{ss}.pth'))
                models1_a[f'{i}'].load_state_dict(states[0])
                models2_a[f'{i}'].load_state_dict(states[0])

            for i in range(1):
                states = torch.load(os.path.join(self.args.log_path, f'path_b{ss}.pth'))
                models1_b[f'{i}'].load_state_dict(states[0])
                models2_b[f'{i}'].load_state_dict(states[0])
            pre_epoch=0


        ema_helper = None
        # 预训练
        step=0
        ema_helper = None



        for epoch in range(start_epoch, start_epoch+pre_epoch):
            data_start = time.time()
            data_time = 0
            for ii, (x, y) in enumerate(train_loader):
                for i in range(self.args.timesteps):
                    n = x.size(0)
                    data_time += time.time() - data_start
                    step += 1

                    b = self.betas

                    #***********************
                    t=torch.tensor([ij for ij in range(0, self.num_timesteps, self.num_timesteps // self.args.timesteps)]).to(self.device)
                    t_index = torch.randint(low=0, high=len(t), size=(n,)).to(self.device)

                    # t_index=torch.ones(n, dtype=torch.long)*int(i)
                    prod=(1 - b).cumprod(dim=0)
                    # print(t,prod.index_select(0, (t)))
                    a = prod.index_select(0, (t)).view(-1, 1).to(self.device)
                    prod1=torch.cat((torch.ones(self.num_timesteps // self.args.timesteps).to(self.device),prod))

                    a_1 = prod1.index_select(0, (t)).view(-1, 1).to(self.device)
                    t1=t.unsqueeze(1).float().clone()
                    t1=t1/self.num_timesteps

                    # ********************



                    pred=models1_a[f'{i}'](t1)
                    loss1 = (10000*(pred-(a/a_1).sqrt())).square()

                    loss1=loss1.sum(dim=(1)).mean(dim=0)

                    if step%21==0:
                        logging.info(
                            f"epoch_a: {epoch}, step: {step}, loss{i}: {loss1.item()},lr{i}:{optimizer1_a[f'{i}'].param_groups[0]['lr']}, data time: {data_time / (i + 1)}"
                        )
                    optimizer1_a[f'{i}'].zero_grad()
                    loss1.backward()
                    optimizer1_a[f'{i}'].step()
                    scheduler1_a[f'{i}'].step(loss1)
                    # if self.config.model.ema:
                    #     ema_helper1.update(model1_path1)
                    data_start = time.time()
            for ii, (x, y) in enumerate(train_loader):
                for i in range(self.args.timesteps):
                    n = x.size(0)
                    data_time += time.time() - data_start
                    step += 1

                    b = self.betas

                    #***********************
                    t=torch.tensor([ij for ij in range(0, self.num_timesteps, self.num_timesteps // self.args.timesteps)]).to(self.device)
                    t_index = torch.randint(low=0, high=len(t), size=(n,)).to(self.device)
                    # t_index=torch.ones(n, dtype=torch.long)*i
                    prod=(1 - b).cumprod(dim=0)

                    a = prod.index_select(0, (t)).view(-1, 1).to(self.device)
                    prod1 = torch.cat((prod, torch.tensor([0]).to(self.device)))
                    a_2 = prod1.index_select(0, (t + self.num_timesteps // self.args.timesteps)).view(-1,
                                                                                                               1).to(
                        self.device)

                    t = t.unsqueeze(1).float() / self.num_timesteps
                    pred = models1_b[f'{i}'](t)

                    loss1 = ((10000 * (((1.0 - a) / (1.0 - a_2)).sqrt() - pred)).square())


                    loss1=loss1.sum(dim=(1)).mean(dim=0)

                    if step%21==0:

                        logging.info(
                            f"epoch_b: {epoch}, step: {step}, loss{i}: {loss1.item()},lr{i}:{optimizer1_b[f'{i}'].param_groups[0]['lr']}, data time: {data_time / (i + 1)}"
                        )
                    optimizer1_b[f'{i}'].zero_grad()
                    loss1.backward()
                    optimizer1_b[f'{i}'].step()
                    scheduler1_b[f'{i}'].step(loss1)
                    # if self.config.model.ema:
                    #     ema_helper1.update(model1_path1)
                    data_start = time.time()

        if not self.args.resume_training:
            for i in range(1):
                states = [
                    models1_a[f'{i}'].state_dict(),
                    optimizer1_a[f'{i}'].state_dict(),
                    scheduler1_a[f'{i}'].state_dict(),
                    epoch,
                    step,
                ]
                torch.save(states, os.path.join(self.args.log_path, f'path_a_pre.pth'))
            for i in range(1):
                states = [
                    models1_b[f'{i}'].state_dict(),
                    optimizer1_b[f'{i}'].state_dict(),
                    scheduler1_b[f'{i}'].state_dict(),
                    epoch,
                    step,
                ]
                torch.save(states, os.path.join(self.args.log_path, f'path_b_pre.pth'))
        if 1:
            model = Model(config)

            # 得到去噪网络的参数
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            elif self.config.data.dataset == 'CELEBA':
                ckpt = './diffusion_models_converted/celeba/ckpt.pth'
                states = torch.load(
                    os.path.join(ckpt),
                    # os.path.join(self.args.log_path, "model-790000.ckpt"),
                    map_location=self.config.device,
                )
                model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

            else:
                raise ValueError


            plot_num = int(len(dataset) // self.config.training.batch_size)

            Lossplot1 = []
            Lossplot2 = []

            step1 = 0
            step2 = 0
        savenum=0
        # torch.autograd.set_detect_anomaly(True)


        for epoch in range(start_epoch+pre_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            Losstemp1 = torch.zeros(plot_num)
            Losstemp2 = torch.zeros(plot_num)
            for flag in FB:

                combined_batches = []
                batch_size=combined_batche_size//config.training.batch_size
                print(batch_size)
                for i, (x, y) in enumerate(train_loader):
                    combined_batches.append((x, y))
                    if (i + 1) % batch_size == 0:
                    # for flag in FB:
                        if flag=='1':

                            for nn in range(1):
                                n = x.size(0)

                                x = x.to(self.device)
                                x = data_transform(self.config, x)
                                e = torch.randn_like(x)
                                b = self.betas

                                loss = self.sample_image_train_path(combined_batche_size,self.device,combined_batches,nn,b, model,
                                                                    models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                    scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                    scheduler2_a,scheduler2_b,1,self.args.timesteps)

                                loss=(sum(loss)/len(loss)).clone()


                                data_time += time.time() - data_start
                                logging.info(
                                    f"Fan!!-epoch: {epoch}, step: {step1}, loss: {loss.item()},lr:{optimizer2_b['3'].param_groups[0]['lr']},data time: {data_time / (i + 1)}"
                                )

                                Losstemp1[step1 % plot_num] = loss.item()
                                step1 += 1

                            if (step1) % savefreq == 0:

                                for i in range(1):
                                    states = [
                                        models2_a[f'{i}'].state_dict(),
                                        optimizer2_a[f'{i}'].state_dict(),
                                        scheduler2_a[f'{i}'].state_dict(),
                                        epoch,
                                        step,
                                    ]
                                    torch.save(states, os.path.join(self.args.log_path, f'path_a_step{savenum}.pth'))
                                for i in range(1):
                                    states = [
                                        models2_b[f'{i}'].state_dict(),
                                        optimizer2_b[f'{i}'].state_dict(),
                                        scheduler2_b[f'{i}'].state_dict(),
                                        epoch,
                                        step,
                                    ]
                                    torch.save(states, os.path.join(self.args.log_path, f'path_b_step{savenum}.pth'))
                                savenum += 1

                            plt.figure(1)
                            Lossplot1.append(torch.mean(Losstemp1[Losstemp1 != 0]));
                            plt.plot(Lossplot2, '.-', label="loss")
                            plt.savefig(os.path.join(self.args.log_path, "Loss_ab1.png"))
                        elif flag=='2':
                            for nn in range(1):
                                n = x.size(0)

                                x = x.to(self.device)
                                x = data_transform(self.config, x)
                                e = torch.randn_like(x)
                                b = self.betas

                                loss = self.sample_image_train_path(combined_batche_size,self.device, combined_batches, nn, b, model,
                                                                    models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                    scheduler1_a, scheduler1_b, models2_a, models2_b,
                                                                    optimizer2_a, optimizer2_b,
                                                                    scheduler2_a, scheduler2_b, 2, self.args.timesteps)
                                loss = (sum(loss) / len(loss)).clone()

                                data_time += time.time() - data_start
                                logging.info(
                                    f"Zheng!!-epoch: {epoch}, step: {step2}, loss: {loss.item()},lr:{optimizer1_b['3'].param_groups[0]['lr']}, data time: {data_time / (i + 1)}"
                                )

                                Losstemp2[step2 % plot_num] = loss.item()

                                data_start = time.time()
                                step2 += 1

                            plt.figure(2)
                            Lossplot2.append(torch.mean(Losstemp2[Losstemp2 != 0]));
                            plt.plot(Lossplot2, '.-', label="loss")
                            plt.savefig(os.path.join(self.args.log_path, "Loss_ab2.png"))


                            if (step2) % savefreq == 0:
                                for i in range(1):
                                    states = [
                                        models1_a[f'{i}'].state_dict(),
                                        optimizer1_a[f'{i}'].state_dict(),
                                        scheduler1_a[f'{i}'].state_dict(),
                                        epoch,
                                        step,
                                    ]
                                    torch.save(states, os.path.join(self.args.log_path, f'path_a_step{savenum}.pth'))
                                for i in range(1):
                                    states = [
                                        models1_b[f'{i}'].state_dict(),
                                        optimizer1_b[f'{i}'].state_dict(),
                                        scheduler1_b[f'{i}'].state_dict(),
                                        epoch,
                                        step,
                                    ]
                                    torch.save(states, os.path.join(self.args.log_path, f'path_b_step{savenum}.pth'))
                                savenum+=1
                        combined_batches = []

    def compute_alpha(self,beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_image_train_path(self,combined_batche_size,device,combined_batches,nn, b,model,models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b,flag=1,timesteps=25, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        loss=[]
        from .denoising import generalized_steps_path1a,generalized_steps_path2a,generalized_steps_path1b,generalized_steps_path2b
        if flag==1:
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                ts = range(0, self.num_timesteps+ skip, skip)
                tl = range(skip, self.num_timesteps + skip, skip)
                ts = np.array(ts) / self.num_timesteps
                tl = np.array(tl) / self.num_timesteps
            loss1=[]
            loss2=[]
            loss2 = generalized_steps_path1b(combined_batche_size,device,combined_batches,nn, self.betas, ts, tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b, self.num_timesteps, eta=self.args.eta)
            loss1 = generalized_steps_path1a(combined_batche_size,device,combined_batches,nn, self.betas, ts, tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b, self.num_timesteps, eta=self.args.eta)
            loss = loss2 + loss1

        elif flag==2:
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                ts = range(self.num_timesteps-skip, -skip , -1*skip)
                tl = range(self.num_timesteps ,0, -1*skip)
                ts = range(0, self.num_timesteps+ skip, skip)
                tl = range(skip, self.num_timesteps + skip, skip)
                ts = np.array(ts) / self.num_timesteps
                tl = np.array(tl) / self.num_timesteps
            loss1 = []
            loss2 = []
            loss2 = generalized_steps_path2b(combined_batche_size,device,combined_batches,nn, self.betas, ts, tl, model, models1_a, models1_b, optimizer1_a, optimizer1_b,
                                                                scheduler1_a, scheduler1_b,models2_a,models2_b,optimizer2_a,optimizer2_b,
                                                                scheduler2_a,scheduler2_b, self.num_timesteps, eta=self.args.eta)
            loss1 = generalized_steps_path2a(combined_batche_size,device, combined_batches, nn, self.betas, ts, tl, model, models1_a,
                                            models1_b, optimizer1_a, optimizer1_b,
                                            scheduler1_a, scheduler1_b, models2_a, models2_b, optimizer2_a,
                                            optimizer2_b,
                                            scheduler2_a, scheduler2_b, self.num_timesteps, eta=self.args.eta)
            loss=loss2+loss1


        # if last:
        #     xs = xs[0][-1]
        # print(xs.shape,x_mid.shape)
        return loss

    def sample_our(self):
        plt.figure()
        for ep in range(0, 30, 2):
            model = Model(self.config)
            if not self.args.use_pretrained:
                if getattr(self.config.sampling, "ckpt_id", None) is None:
                    states = torch.load(
                        os.path.join('./exp/logs/celeba-FM-notcenter/', "ckpt.pth"),
                        map_location=self.config.device,
                    )
                else:
                    states = torch.load(
                        os.path.join(
                            self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                        ),
                        map_location=self.config.device,
                    )
                model = model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                    ckpt = get_ckpt_path(f"ema_{name}")
                    print("Loading checkpoint {}".format(ckpt))
                    model.load_state_dict(torch.load(ckpt, map_location=self.device))
                    model.to(self.device)
                    model = torch.nn.DataParallel(model)

                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                    ckpt = get_ckpt_path(f"ema_{name}")
                    print("Loading checkpoint {}".format(ckpt))
                    model.to(self.device)
                    model.load_state_dict(torch.load(ckpt, map_location=self.device))
                    model.to(self.device)
                    model = torch.nn.DataParallel(model)

                elif self.config.data.dataset == 'CELEBA':
                    ckpt = './diffusion_models_converted/celeba/ckpt.pth'
                    states = torch.load(
                        os.path.join(ckpt),
                        # os.path.join(self.args.log_path, "model-790000.ckpt"),
                        map_location=self.config.device,
                    )
                    model.to(self.device)
                    model = torch.nn.DataParallel(model)
                    model.load_state_dict(states[0], strict=True)

                else:
                    raise ValueError

            model.eval()
            config = self.config


            seed = 0
            # 设置全局随机数种子
            torch.manual_seed(seed)

            # 如果使用GPU，也设置GPU上的随机数种子
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # if (ep//5)%2==1:
            #     continue
            # if (ep+1)%5!=0:
            #     continue
            final = 1
            ss = f'_step{ep}'
            if final==0:
                if not os.path.exists(os.path.join(self.args.log_path, f'path_a{ss}.pth')):
                    continue
                # while not os.path.exists(os.path.join(self.args.log_path, f'path_a{ss}.pth')):
                #     time.sleep(10)
            elif final==1:
                if not os.path.exists(os.path.join(self.args.log_path, f'(final)path_a{ss}.pth')):
                    continue
            print(ep)

            models2_a = {}
            models2_b = {}

            model_type=self.model_type
            if model_type == 0:
                model_patha = KAN(self.stru, grid_size=self.grid_num*self.args.timesteps, grid_range=[0, 1]).to(self.device)
                model_pathb = KAN(self.stru, grid_size=self.grid_num*self.args.timesteps, grid_range=[0, 1]).to(self.device)
            elif model_type == 1:
                model_patha = MLP_path(config).to(self.device)
                model_pathb = MLP_path(config).to(self.device)
                model_patha.initialize()
                model_pathb.initialize()

            import math

            for i in range(self.args.timesteps):
                models2_a[f'{i}'] = model_patha
                models2_a[f'{i}'].to(self.device)
                models2_b[f'{i}'] = model_pathb
                models2_b[f'{i}'].to(self.device)

            for i in range(1):
                # ss='_pre'
                # ss=''
                if final == 1:
                    states = torch.load(os.path.join(self.args.log_path, f'(final)path_a{ss}.pth'))
                    models2_a[f'{i}'].load_state_dict(states[0])
                    states = torch.load(os.path.join(self.args.log_path, f'(final)path_b{ss}.pth'))
                    models2_b[f'{i}'].load_state_dict(states[0])
                elif final==0:
                    states = torch.load(os.path.join(self.args.log_path, f'path_a{ss}.pth'))
                    models2_a[f'{i}'].load_state_dict(states[0])
                    states = torch.load(os.path.join(self.args.log_path, f'path_b{ss}.pth'))
                    models2_b[f'{i}'].load_state_dict(states[0])
                models2_a[f'{i}'].eval()
                models2_b[f'{i}'].eval()
                print(states[4])

            n = config.sampling.batch_size
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            a,b = self.print_test_path(x, model,models2_a,models2_b,timesteps=self.args.timesteps)

            a=np.array(torch.tensor(a).cpu())
            b=np.array(torch.tensor(b).cpu())
            print(a,b)

            import shutil
            shutil.rmtree(self.args.image_folder)
            os.makedirs(self.args.image_folder)
            # print(self.args.image_folder)

            if self.args.fid:
                self.sample_fid_our(a,b,model,models2_a,models2_b)
            elif self.args.interpolation:
                self.sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedeure not defined")
            del model
            torch.cuda.empty_cache()
            root1 = "./exp/datasets/cifar10_test/cifar-10-batches-py/output"
            root2 = self.args.image_folder
            print(root2)
            fid = self.FID(root1, root2)
            # with open(os.path.join(self.args.log_path, 'FID-church.txt'), 'a') as log_file:
            with open(os.path.join(self.args.log_path, 'FID-bedroom.txt'), 'a') as log_file:
                log_file.write(f'a:{a} .\n b: {b}.\n')
                log_file.write(f'Epoch {ep} FID: {fid}.\n')
            print(f'Epoch {ep} FID: {fid}.\n')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Variation of a and b with n')
        plt.legend()
        plt.savefig(os.path.join(self.args.log_path, 'ab_variation.png'))  # 保存图像
        plt.show()





    def FID(self,root1,root2):

        flag=0
        if flag ==1:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader

            # 定义数据转换
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 加载数据集
            real_dataset = datasets.ImageFolder(root1, transform=transform)
            fake_dataset = datasets.ImageFolder(root1, transform=transform)

            real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
            fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

            # 初始化FID计算器
            fid = FrechetInceptionDistance(feature=2048)

            # 计算真实图像的特征
            for images, _ in real_loader:
                fid.update(images, real=True)

            # 计算生成图像的特征
            for images, _ in fake_loader:
                fid.update(images, real=False)

            # 计算FID分数
            f = fid.compute()
            print(f"FID score: {f.item()}")

            return f.item()
        elif flag==0:
            import subprocess


            # 使用 subprocess 运行 pytorch-fid 命令并捕获输出
            result = subprocess.run(['python', '-m', 'pytorch_fid', root1, root2], stdout=subprocess.PIPE)

            # 解析输出并赋值给变量 f
            output = result.stdout.decode('utf-8')
            # f = float(output.strip().split()[-1])
            # print(f"FID score: {f}")
            parts = output.strip().split()

            if len(parts) == 0:
                print(output)
                print("Error: No output returned.")
                return -1
            else:
                try:
                    f = float(parts[-1])
                    print(f"FID score: {f}")
                except ValueError:
                    print("Error: Unable to convert output to float.")
            return f


    def sample_fid_our(self,a,b, model,models2_a,models2_b):
        config = self.config

        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image_test_path(a,b,x, model,models2_a,models2_b,timesteps=self.args.timesteps)
                x = inverse_data_transform(config, x)
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image_test_path(self,a,b, x, model,models2_a,models2_b,timesteps=1000, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
                # seq = range(0, t*skip, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from .denoising import generalized_steps_path_test

            x = generalized_steps_path_test(a,b,x, seq, model, models2_a,models2_b, self.betas, eta=self.args.eta)

        if last:
            x = x[0][-1]
            # print(x.shape)
        return x

    def print_test_path(self, x, model,models2_a,models2_b,timesteps=1000, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
                # seq = range(0, t*skip, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from .denoising import print_testpath

            a,b = print_testpath(x, seq, model, models2_a,models2_b, self.betas, eta=self.args.eta)
        return a,b


    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from .denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from .denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
