

from lib.networks.base_network import MsImageDis
from lib.networks.shadow_network  import Gen
from lib.utils.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim,MedianPool2d
from lib.models.base_model import BaseModels

from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import sys
from torchvision.utils import save_image

# x y is subject
# a b is illumination

class Models(BaseModels):
    def __init__(self, hyperparameters):
        super(Models, self).__init__()
        lr = hyperparameters['lr']
        self.model_name = hyperparameters['models_name']
        # Initiate the networks

        if(self.model_name=='shadow'):
            self.gen = Gen( hyperparameters['gen'])
        else:
            sys.exit('error on models')

        self.gen = nn.DataParallel(self.gen)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']



        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters()) #+ list(self.gen_b.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        self.dis_scheduler = None

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))


        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg = nn.DataParallel(self.vgg)






    def gen_update(self,  p_,mask_,light,shadow,hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        out_x = self.gen.forward(torch.mul(p_, mask_),light)

        # main relight loss
        self.loss_gen_prime_x_b = self.recon_criterion_mask(out_x, shadow, mask_)
        self.loss_gen_total = hyperparameters['relight'] * self.loss_gen_prime_x_b
        self.loss_gen_total.backward()
        self.gen_opt.step()


    def dis_update(self, x_a,gt_xb,x_mask, gt_x_p,hyperparameters):
        self.dis_opt.zero_grad()
        out_x,_ = self.gen.forward(torch.mul(x_a, x_mask))
        # D loss
        out_x = torch.mul(out_x, x_mask)
        gt_xb = torch.mul(gt_xb, x_mask)
        self.loss_dis = self.dis.calc_dis_loss(out_x.detach(), gt_xb)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()











