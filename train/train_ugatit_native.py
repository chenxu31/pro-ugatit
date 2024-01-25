import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ugatit_pro import UGATIT
from configs.cfgs_ugatit_native import cfgs
import torch
import itertools
from utils.utils import AvgMeter, init_path, get_scheduler

from datetime import datetime
from tqdm import tqdm
from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader
from models.image_pool import ImagePool
from models.loss import GANLoss, CAMLoss
import numpy as np
import numpy
import argparse
from main_utils import *
from PIL import Image, ImageFont, ImageDraw
import platform
import skimage.io
import pdb

if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_cmf_pt as common_cmf
import common_pelvic_pt as common_pelvic



class Train:
    def __init__(self, device, model, args):
        self.device = device
        self.model = model
        self.args = args

        self.client = None
        
        self.messageChannel = None

        if args.dataset == "pelvic":
            self.common_file = common_pelvic
            self.dataset_s = common_pelvic.Dataset(args.data_dir, "ct", n_slices=args.inc, debug=args.debug)
            self.dataset_t = common_pelvic.Dataset(args.data_dir, "cbct", n_slices=args.inc, debug=args.debug)

            self.data_loader_s = torch.utils.data.DataLoader(self.dataset_s,
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=NUM_WORKERS,
                                                             pin_memory=True,
                                                             drop_last=True)
            self.data_loader_t = torch.utils.data.DataLoader(self.dataset_t,
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=NUM_WORKERS,
                                                             pin_memory=True,
                                                             drop_last=True)

            self.val_data_s, self.val_data_t, _, _ = common_pelvic.load_val_data(args.data_dir, valid=True)
        elif args.dataset == "cmf":
            self.common_file = common_cmf
            self.dataset_s = common_cmf.Dataset(args.data_dir, "ct", n_slices=args.inc, debug=args.debug)
            self.dataset_t = common_cmf.Dataset(args.data_dir, "mri", n_slices=args.inc, debug=args.debug)

            self.data_loader_s = torch.utils.data.DataLoader(self.dataset_s,
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=NUM_WORKERS,
                                                             pin_memory=True,
                                                             drop_last=True)
            self.data_loader_t = torch.utils.data.DataLoader(self.dataset_t,
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=NUM_WORKERS,
                                                             pin_memory=True,
                                                             drop_last=True)

            self.val_data_s, self.val_data_t, _ = common_cmf.load_test_data(args.data_dir)
        else:
            assert 0

        self.init()
        if len(args.resume) != 0:  # resume
            print('start load weight for resuming training.')
            self.model.load_state_dict(args.resume)

    def init(self):
        opt = self.args
        
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)
        self.crit_cycle = torch.nn.L1Loss()
        self.crit_idt = torch.nn.L1Loss()
        self.crit_gan = GANLoss(opt.gan_mode).cuda()
        self.cam_loss = CAMLoss()
        self.optim_G = torch.optim.Adam(itertools.chain(self.model.G_A.parameters(),
                                                        self.model.G_B.parameters()),
                                        lr=opt.lr,
                                        betas=(opt.beta1, 0.999))
        self.optim_D = torch.optim.Adam(itertools.chain(self.model.D_A.parameters(),
                                                        self.model.D_B.parameters()),
                                        lr=opt.lr, betas=(opt.beta1, 0.999))  # default: 0.5
        self.optimizers = [self.optim_G, self.optim_D]

        self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in self.optimizers]


    def update_G(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B, cam_ab, cam_ba = inp
        opt = self.args
        lambda_idt = opt.lambda_identity  # G:x->y; min|G(y)-y|
        if lambda_idt > 0:  # identity loss
            idt_A, cam_bb = self.model.G_A(real_B)
            idt_B, cam_aa = self.model.G_B(real_A)
            loss_itd_A = self.crit_idt(idt_A, real_B) * lambda_idt
            loss_itd_B = self.crit_idt(idt_B, real_A) * lambda_idt
        else:
            loss_itd_A = 0
            loss_itd_B = 0

        l_fake_B_logits, l_fake_B_cam_logits, \
        g_fake_B_logits, g_fake_B_cam_logits = self.model.D_A(fake_B)  # local and global

        l_fake_A_logits, l_fake_A_cam_logits, \
        g_fake_A_logits, g_fake_A_cam_logits = self.model.D_B(fake_A)
        # gan loss
        loss_G_A = self.crit_gan(l_fake_B_logits, True) + self.crit_gan(l_fake_B_cam_logits, True) + \
            self.crit_gan(g_fake_B_logits, True) + self.crit_gan(g_fake_B_cam_logits, True)
        loss_G_B = self.crit_gan(l_fake_A_logits, True) + self.crit_gan(l_fake_A_cam_logits, True) + \
            self.crit_gan(g_fake_A_logits, True) + self.crit_gan(g_fake_A_cam_logits, True)
        # cycle loss
        loss_cycle_A = self.crit_cycle(rec_A, real_A) * opt.lambda_cycle
        loss_cycle_B = self.crit_cycle(rec_B, real_B) * opt.lambda_cycle
        # cam loss
        if lambda_idt > 0:
            cam_loss_A = self.cam_loss(cam_ab, cam_bb) * opt.lambda_cam
            cam_loss_B = self.cam_loss(cam_ba, cam_aa) * opt.lambda_cam
        else:
            cam_loss_A = 0
            cam_loss_B = 0

        loss_A = loss_G_A + loss_cycle_A + cam_loss_A + loss_itd_A  # G_A的loss
        loss_B = loss_G_B + loss_cycle_B + cam_loss_B + loss_itd_B  # G_B的loss

        loss_G = loss_A + loss_B
        loss_G.backward()
        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, cam_loss_A, cam_loss_B

    def update_D_basic(self, netD, real, fake):
        l_real_logits, l_real_cam_logits, \
        g_real_logits, g_real_cam_logits = netD(real)

        l_fake_logits, l_fake_cam_logits, \
        g_fake_logits, g_fake_cam_logits = netD(fake)

        loss_real = self.crit_gan(l_real_logits, True) + self.crit_gan(g_real_logits, True)
        loss_real_cam = self.crit_gan(l_real_cam_logits, True) + self.crit_gan(g_real_cam_logits, True)
        loss_fake = self.crit_gan(l_fake_logits, False) + self.crit_gan(g_fake_logits, False)
        loss_fake_cam = self.crit_gan(l_fake_cam_logits, False) + self.crit_gan(g_fake_cam_logits, False)
        return loss_fake, loss_real, loss_fake_cam, loss_real_cam

    def update_D(self, inp):
        real_A, real_B, fake_A, fake_B, rec_A, rec_B, _, _ = inp
        fake_B = self.fake_B_pool.query(fake_B)
        loss_fake, loss_real, loss_fake_cam, loss_real_cam \
            = self.update_D_basic(self.model.D_A, real_B, fake_B)
        loss_D_A = loss_fake + loss_real + loss_fake_cam + loss_real_cam
        loss_D_A.backward()

        fake_A = self.fake_A_pool.query(fake_A)
        loss_fake, loss_real, loss_fake_cam, loss_real_cam \
            = self.update_D_basic(self.model.D_B, real_A, fake_A)
        loss_D_B = loss_fake + loss_real + loss_fake_cam + loss_real_cam
        loss_D_B.backward()
        return loss_D_A, loss_D_B

    def train_on_step(self, inp):
        inp = self.model(inp)
        self.realA = inp[0]  # b,c,h,w;  for tensorboard
        self.realB = inp[1]
        self.fakeA = inp[2]
        self.fakeB = inp[3]
        # 先更新G
        self.set_requires_grad([self.model.D_A, self.model.D_B], False)
        self.optim_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_cam_A, loss_cam_B = self.update_G(inp)
        self.optim_G.step()

        self.set_requires_grad([self.model.D_A, self.model.D_B], True)
        self.optim_D.zero_grad()
        loss_D_A, loss_D_B = self.update_D(inp)
        self.optim_D.step()

        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, \
               loss_D_A, loss_D_B, loss_cam_A, loss_cam_B

    def train_on_epoch(self, epoch):
            loss_meters = []
            loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'D_A', 'D_B', 'CAM_A', 'CAM_B']
            for _ in range(len(loss_names)):
                loss_meters.append(AvgMeter())

            for iter_idx, (data_s, data_t) in enumerate(zip(self.data_loader_s, self.data_loader_t)):

                if (data_s["image"].contiguous().view(data_s["image"].shape[0], -1).max(1)[0].min() <= -1 or
                        data_t["image"].contiguous().view(data_t["image"].shape[0], -1).max(1)[0].min() <= -1):
                    continue

                inp = {
                    "A": data_s["image"],
                    "B": data_t["image"],
                }
                
                losses_set = self.train_on_step(inp)
                
                for loss, meter in zip(losses_set, loss_meters):
                    meter.update(loss)

    def train(self):
            
            args = self.args


            if args.start_epoch != 0:  # if resume, changing lr.
                for _ in range(args.start_epoch):
                    self.update_learning_rate()

            best_psnr = 0
            for epoch in range(args.start_epoch, args.total_epoch):
                self.train_on_epoch(epoch)

                self.update_learning_rate()

                self.model.G_B.eval()

                patch_shape = (args.inc, self.val_data_s[0].shape[1], self.val_data_s[0].shape[2])
                psnr_list = np.zeros((len(self.val_data_s),), np.float32)
                with torch.no_grad():
                    for i in range(len(self.val_data_s)):
                        syn_im = common_net.produce_results(self.device, self.model.G_B, [patch_shape, ],
                                                            [self.val_data_t[i], ],
                                                            data_shape=self.val_data_t[i].shape, patch_shape=patch_shape,
                                                            is_seg=False, batch_size=16)
                        psnr_list[i] = common_metrics.psnr(syn_im, self.val_data_s[i])

                self.model.G_B.train()

                print("%s  Epoch:%d  Val psnr:%f/%f" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, psnr_list.mean(), psnr_list.std()))

                if psnr_list.mean() > best_psnr:
                    best_psnr = psnr_list.mean()
                    torch.save(self.model.state_dict(), os.path.join(args.result_dir, args.dataset, 'train_best.pt'))

                torch.save(self.model.state_dict(), os.path.join(args.result_dir, args.dataset, 'train_last.pt'))
            

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def main_worker(device, args):
    # init model and dataloader
    
    model = UGATIT(args)
    model.train()
    if args.gpu >= 0:
        model.cuda()

    trainer = Train(device, model, args)
    trainer.train()

if __name__ == '__main__':
    args = handleParser()
    args.__dict__.update(cfgs)
    #pdb.set_trace()
    #args = argFixer(args,commandArgs,args.data_dir,args.data_dir)
    #print(args)

    output_dir = os.path.join(args.result_dir, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main_worker(device, args)