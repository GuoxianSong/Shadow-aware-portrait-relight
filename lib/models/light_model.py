
from lib.networks.base_network import MsImageDis
from lib.networks.light_network  import Gen
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

        if(self.model_name=='light'):
            self.gen = Gen(hyperparameters['input_dim_a'], hyperparameters['gen'])
        else:
            sys.exit('error on models')

        self.gen = nn.DataParallel(self.gen)
        #self.gen = self.gen.to('cuda')




        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']



        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim*2, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim*2, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters()) #+ list(self.gen_b.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)


        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False



    def recon_criterion_rmse(self,input,target):
        out=0
        psnr_=0
        ssim_=0
        if(len(input.shape)==3):
            tmp=torch.mean((input - target)**2)
            tmp = tmp**0.5
            psnr = 20*torch.log10(1/tmp)
            img1 = torch.unsqueeze(input,dim=0)
            img2 = torch.unsqueeze(target,dim=0)
            ssim_loss=ssim(img1, img2)
            #ssim_loss = pytorch_ssim.SSIM(window_size=11)
            return tmp.item(),psnr.item(),ssim_loss.item()
        else:
            for i in range(len(input)):
                tmp = torch.mean((input[i] - target[i]) ** 2)
                tmp=tmp** 0.5
                out+=tmp
                psnr_+=20*torch.log10(1/tmp)
                img1 = torch.unsqueeze(input[i], dim=0)
                img2 = torch.unsqueeze(target[i], dim=0)
                ssim_ +=ssim(img1, img2)

            return (out/len(input)).item(),(psnr_/len(input)).item(),(ssim_/len(input)).item()




    def gen_update(self, beauty,label,weight,hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        out_x = self.gen.forward(beauty)

        # main relight loss
        self.loss_gen_prime_x_b = self.recon_criterion(out_x, label) + self.recon_criterion_mask(out_x, label,weight)

        # GAN loss
        self.loss_gen_total = hyperparameters['relight'] * self.loss_gen_prime_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()


        image_recons = out_x[0:1].detach().cpu()[:3]
        image_gt = label[0:1].detach().cpu()[:3]

        self.image_display = torch.cat((image_recons, image_gt),dim=3)
        self.image_input = beauty[0:1].detach().cpu()[:3]

    def dis_update(self,  beauty,label, hyperparameters):
        self.dis_opt.zero_grad()
        out_x = self.gen.forward(beauty)
        # D loss
        self.loss_dis = self.calc_dis_loss(self.dis.forward(out_x.detach()),self.dis.forward(label)) #self.dis.calc_dis_loss(out_x.detach(), label)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters,need_opt=True,path=None):
        # Load generators
        if(path==None):
            last_model_name = get_model_list(checkpoint_dir, "gen")
        else:
            last_model_name=path
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        if(need_opt):
            self.gen_opt.load_state_dict(state_dict['gen_opt'])
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        torch.save({'gen': self.gen.state_dict(),'gen_opt': self.gen_opt.state_dict()}, gen_name)







