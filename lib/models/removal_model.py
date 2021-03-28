

from lib.networks.base_network import MsImageDis
from lib.networks.removal_network import Gen
from lib.utils.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim
from lib.models.base_model import BaseModels

from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import sys


# x y is subject
# a b is illumination

class Models(BaseModels):
    def __init__(self, hyperparameters):
        super(Models, self).__init__()
        lr = hyperparameters['lr']
        self.model_name = hyperparameters['models_name']
        # Initiate the networks

        if(self.model_name=='removal'):
            self.gen = Gen(hyperparameters['input_dim_a'], hyperparameters['gen'])
            self.dis = MsImageDis(hyperparameters['input_dim_a'],
                                    hyperparameters['dis'])  # discriminator for domain a
        else:
            sys.exit('error on models')

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        self.gen = nn.DataParallel(self.gen)
        self.dis = nn.DataParallel(self.dis)

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        dis_params = list(self.dis.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)


        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.vgg = nn.DataParallel(self.vgg)


    def gen_update(self, out_data,hyperparameters):
        Xa_out, X_removal, mask, depth=out_data

        self.gen_opt.zero_grad()
        # encode
        out_x,p_x = self.gen.forward(torch.mul(Xa_out, mask))
        # main relight loss
        self.loss_gen_prime_x_b = self.recon_criterion_mask(out_x, X_removal, mask)

        # main p loss
        self.loss_gen_prime_x_p = self.recon_criterion_mask(p_x, depth, mask[:,0:1,:,:])

        # GAN loss
        self.loss_gen_adv = self.calc_gen_loss(self.dis.forward(torch.mul(out_x, mask)))

        self.loss_gen_total = hyperparameters['relight'] * self.loss_gen_prime_x_b + \
                                hyperparameters['x_p'] * self.loss_gen_prime_x_p + hyperparameters['gan_w']* self.loss_gen_adv

        self.loss_gen_total.backward()
        self.gen_opt.step()

        image_anchor = Xa_out[0:1].detach().cpu()[:3]
        image_recons = torch.mul(out_x, mask)[0:1].detach().cpu()[:3]
        image_gt = X_removal[0:1].detach().cpu()[:3]
        depth_gt = (depth[0:1].detach().cpu()[:3]).repeat(1,3,1,1)
        depth = (p_x[0:1].detach().cpu()[:3]).repeat(1,3,1,1)
        self.image_display = torch.cat((image_anchor, image_recons, image_gt,depth_gt,depth),dim=3)

    def dis_update(self, x_a,gt_xb,x_mask,hyperparameters):
        self.dis_opt.zero_grad()
        out_x,_ = self.gen.forward(torch.mul(x_a, x_mask))
        # D loss
        out_x = torch.mul(out_x, x_mask)
        gt_xb = torch.mul(gt_xb, x_mask)
        self.loss_dis = self.calc_dis_loss(self.dis.forward(out_x.detach()),self.dis.forward(gt_xb))
        #self.loss_dis = self.dis.calc_dis_loss(out_x.detach(), gt_xb)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()











