

from lib.networks.base_network import MsImageDis
from lib.networks.render_network import Gen
from lib.utils.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim,MedianPool2d
from lib.models.base_model import BaseModels

from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F

# x y is subject
# a b is illumination

class Models(BaseModels):
    def __init__(self, hyperparameters):
        super(Models, self).__init__()
        lr = hyperparameters['lr']
        self.model_name = hyperparameters['models_name']
        # Initiate the networks

        if(self.model_name=='render'):
            self.gen = Gen( hyperparameters['gen'])
            self.dis = MsImageDis(hyperparameters['dis']['input_dim_dis'],
                                    hyperparameters['dis'])  # discriminator for domain a
        else:
            sys.exit('error on models')

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']



        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])

        self.gen = nn.DataParallel(self.gen)
        self.dis = nn.DataParallel(self.dis)


        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters()) #+ list(self.gen_b.parameters())

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





    def gen_update(self,out_data,hyperparameters):
        X_removal, mask, depth, Xb_out, b_light, b_light_label, b_shadow =out_data
        self.gen_opt.zero_grad()
        # encode
        b_out_x = self.gen.forward(torch.mul(X_removal, mask), torch.mul(b_shadow, mask[:, 0:1, :, :]), b_light_label)
        b_out_x = torch.mul(b_out_x, mask) +  torch.mul(Xb_out, 1-mask)
        # main relight loss
        self.loss_gen_prime_x_b = self.recon_criterion_mask(Xb_out, b_out_x, mask)

        # GAN loss
        self.loss_gen_adv = self.calc_gen_loss(self.dis.forward(b_out_x))

        # domain-invariant perceptual loss
        self.loss_gen_vgg =self.compute_vgg_loss(self.vgg, torch.mul(Xb_out, mask), torch.mul(b_out_x, mask)) \
                            if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_total = hyperparameters['relight'] * self.loss_gen_prime_x_b +\
                                hyperparameters['vgg_w'] * self.loss_gen_vgg + \
                                hyperparameters['gan_w'] * self.loss_gen_adv
        self.loss_gen_total.backward()
        self.gen_opt.step()


    def dis_update(self, x_a,gt_xb,x_mask,light,shadow,hyperparameters):
        self.dis_opt.zero_grad()
        out_x = self.gen.forward(torch.mul(x_a, x_mask),torch.mul(shadow, x_mask[:,0:1,:,:]),light)
        # D loss
        out_x= torch.mul(out_x, x_mask) +  torch.mul(gt_xb, 1-x_mask)
        self.loss_dis = self.calc_dis_loss(self.dis.forward(out_x.detach()), self.dis.forward(gt_xb))

        #self.loss_dis = self.dis.calc_dis_loss(out_x.detach(), gt_xb)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()







