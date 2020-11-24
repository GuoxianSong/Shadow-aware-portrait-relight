

from lib.networks.base_network import MsImageDis
import lib.networks.render_network
import lib.networks.removal_network
import lib.networks.light_network
import lib.networks.shadow_network

from lib.utils.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim,get_config,MedianPool2d
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
    def __init__(self, hyperparameters,live=False):
        super(Models, self).__init__()
        lr = hyperparameters['lr']
        self.model_name = hyperparameters['models_name']
        # Initiate the networks

        if(self.model_name=='end2end'):
            light_config = get_config('configs/light.yaml')
            removal_config = get_config('configs/removal.yaml')
            render_config = get_config('configs/render.yaml')
            shadow_config = get_config('configs/shadow.yaml')

            #removal hyperparameters['input_dim_a'], hyperparameters['gen']
            self.removal_gen = lib.networks.removal_network.Gen(removal_config['input_dim_a'], removal_config['gen'],live=live)

            #light
            self.light_gen = lib.networks.light_network.Gen(light_config['input_dim_a'], light_config['gen'])


            #shadow
            self.shadow_gen = lib.networks.shadow_network.Gen(shadow_config['gen'])

            #render
            self.render_gen = lib.networks.render_network.Gen(render_config['gen'])
            self.render_dis = MsImageDis(render_config['dis']['input_dim_dis'],
                                         render_config['dis'])  # discriminator for domain a
        else:
            sys.exit('error on models')

        #
        self.removal_gen = nn.DataParallel(self.removal_gen)
        self.light_gen = nn.DataParallel(self.light_gen)
        self.shadow_gen = nn.DataParallel(self.shadow_gen)
        self.render_gen = nn.DataParallel(self.render_gen)
        self.render_dis = nn.DataParallel(self.render_dis)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.removal_gen.parameters()) + list(self.light_gen.parameters()) + \
                     list(self.shadow_gen.parameters()) + list(self.render_gen.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        dis_params = list(self.render_dis.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.dis_scheduler=None

        # Network weight initialization

        if(0):
            self.initialization()
        else:
            self.apply(weights_init(hyperparameters['init']))
            self.render_dis.apply(weights_init('gaussian'))

        # Load VGG model if needed
        self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg = nn.DataParallel(self.vgg)


    def initialization(self):

        last_model_name = get_model_list('outputs/light/checkpoints', "gen")
        state_dict = torch.load(last_model_name)
        self.light_gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        print('light Resume from iteration %d' % iterations)


        last_model_name = get_model_list('outputs/removal/checkpoints', "gen")
        state_dict = torch.load(last_model_name)
        self.removal_gen.module.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        print('removal Resume from iteration %d' % iterations)

        last_model_name = get_model_list('outputs/shadow/checkpoints', "gen")
        state_dict = torch.load(last_model_name)
        self.shadow_gen.module.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        print('shadow Resume from iteration %d' % iterations)

        last_model_name = get_model_list('outputs/render/checkpoints', "gen")
        state_dict = torch.load(last_model_name)
        self.render_gen.module.load_state_dict(state_dict['gen'])
        self.render_dis.module.load_state_dict(state_dict['dis'])
        iterations = int(last_model_name[-11:-3])
        print('render Resume from iteration %d' % iterations)



    def gen_update(self, out_data,hyperparameters):
        Xa_out, X_removal, mask, depth, Xb_out, b_light, b_light_label, b_shadow=out_data

        self.gen_opt.zero_grad()
        removal_x, depth_x = self.removal_gen.forward(torch.mul(Xa_out, mask))

        #relight
        infer_b_light = self.light_gen.forward(b_light)
        infer_b_shaodw = self.shadow_gen.forward(torch.mul(depth_x, mask[:, 0:1, :, :]), infer_b_light)
        infer_b = self.render_gen.forward(torch.mul(removal_x, mask), torch.mul(infer_b_shaodw, mask[:, 0:1, :, :]),
                                          infer_b_light)

        infer_b = torch.mul(infer_b, mask) +torch.mul(1-infer_b, Xb_out)

        # main relight loss
        self.loss_end2end = self.recon_criterion_mask(infer_b, Xb_out, mask)


        # GAN loss
        self.loss_end2end_adv = self.calc_gen_loss(self.render_dis.forward(infer_b))


        # domain-invariant perceptual loss
        self.loss_end2end_vgg = self.compute_vgg_loss(self.vgg, torch.mul(infer_b, mask), torch.mul(Xb_out, mask)) \
          if hyperparameters['vgg_w'] > 0 else 0

        #removal
        self.loss_removal_image = self.recon_criterion_mask(removal_x, X_removal, mask)
        self.loss_removal_depth = self.recon_criterion_mask(depth_x, depth, mask[:,0:1,:,:])

        #light
        self.loss_light =  self.recon_criterion(infer_b_light,b_light_label)

        #shadow
        self.loss_shadow = self.recon_criterion_mask(infer_b_shaodw, b_shadow, mask[:,0:1,:,:])


        self.loss_gen_total = hyperparameters['relight'] * self.loss_end2end + \
                              hyperparameters['removal'] * self.loss_removal_image + \
                              hyperparameters['depth'] * self.loss_removal_depth + \
                              hyperparameters['light'] * self.loss_light + \
                              hyperparameters['shadow'] *self.loss_shadow + \
                              hyperparameters['vgg_w'] * self.loss_end2end_vgg + \
                              hyperparameters['gan_w'] * self.loss_end2end_adv

        self.loss_gen_total.backward()
        self.gen_opt.step()


    def dis_update(self, Xa_out,X_removal,Xb_out,mask,depth,shadow,light,light_weight,light_label,hyperparameters):
        self.dis_opt.zero_grad()
        # removal
        removal_x, depth_x = self.removal_gen.forward(torch.mul(Xa_out, mask))

        # light
        out_light = self.light_gen.forward(light)

        # shadow
        shadow_x = self.shadow_gen.forward(torch.mul(depth_x, mask[:, 0:1, :, :]), out_light)

        # render
        out_x = self.render_gen.forward(torch.mul(removal_x, mask), torch.mul(shadow_x, mask[:, 0:1, :, :]), out_light)

        # D loss
        out_x= torch.mul(out_x, mask) +  torch.mul(Xb_out, 1-mask)


        self.loss_dis = self.calc_dis_loss(self.render_dis.forward(out_x.detach()),self.render_dis.forward(Xb_out))
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()


    def resume(self, checkpoint_dir, hyperparameters,need_opt=True,path=None):
        # Load generators
        if(path==None):
            last_model_name = get_model_list(checkpoint_dir, "gen")
        else:
            last_model_name=path
        state_dict = torch.load(last_model_name)
        self.removal_gen.module.load_state_dict(state_dict['removal_gen'])
        self.light_gen.module.load_state_dict(state_dict['light_gen'])
        self.shadow_gen.module.load_state_dict(state_dict['shadow_gen'])
        self.render_gen.module.load_state_dict(state_dict['render_gen'])
        self.render_dis.module.load_state_dict(state_dict['dis'])
        iterations = int(last_model_name[-11:-3])
        if(need_opt):
            self.gen_opt.load_state_dict(state_dict['gen_opt'])
            self.dis_opt.load_state_dict(state_dict['dis_opt'])
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        torch.save({'removal_gen': self.removal_gen.module.state_dict(),'light_gen': self.light_gen.module.state_dict(),
                    'shadow_gen': self.shadow_gen.module.state_dict(),'render_gen': self.render_gen.module.state_dict(),
                    'gen_opt': self.gen_opt.state_dict(),'dis':self.render_dis.module.state_dict(),
                    'dis_opt':self.dis_opt.state_dict()}, gen_name)

