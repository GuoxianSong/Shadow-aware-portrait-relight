
from lib.utils.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim
from torch.autograd import Variable
import torch
import torch.nn as nn

import os

# x y is subject
# a b is illumination

class BaseModels(nn.Module):
    def __init__(self):
        super(BaseModels, self).__init__()


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_mask(self, input, target,mask):
        return torch.mean(torch.abs(torch.mul(input,mask) - torch.mul(target,mask)))

    def recon_criterion_rmse(self,input,target,mask,denorm=True):
        if(denorm):
            input = (input*0.5+0.5)
            target= (target*0.5+0.5)
        out=0
        psnr_=0
        ssim_=0
        if(len(input.shape)==3):
            tmp=torch.sum((torch.mul(input, mask) - torch.mul(target, mask))**2)
            tmp/=torch.sum(mask)
            tmp = tmp**0.5
            psnr = 20*torch.log10(1/tmp)
            img1 = torch.mul(input, mask) + torch.mul(target, 1-mask)
            img1 = torch.unsqueeze(img1,dim=0)
            img2 = torch.unsqueeze(target,dim=0)
            ssim_loss=ssim(img1, img2)
            #ssim_loss = pytorch_ssim.SSIM(window_size=11)
            return tmp.item(),psnr.item(),ssim_loss.item()
        else:
            for i in range(len(input)):
                tmp=torch.sum((torch.mul(input[i], mask[i]) - torch.mul(target[i], mask[i]))**2)
                tmp/=torch.sum(mask[i])
                tmp=tmp** 0.5
                out+=tmp
                psnr_+=20*torch.log10(1/tmp)

                img1 = torch.mul(input[i], mask[i]) + torch.mul(target[i], 1 - mask[i])
                img1 = torch.unsqueeze(img1, dim=0)
                img2 = torch.unsqueeze(target[i], dim=0)
                ssim_ +=ssim(img1, img2)

            return (out/len(input)).item(),(psnr_/len(input)).item(),(ssim_/len(input)).item()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


    def test_recon_criterion_mask(self, input, target,mask):
        out=0
        if(len(input.shape)==3):
            tmp=torch.sum(torch.abs(torch.mul(input, mask) - torch.mul(target, mask)))
            tmp/=torch.sum(mask)
            return tmp
        else:
            for i in range(len(input)):
                tmp=torch.sum(torch.abs(torch.mul(input[i], mask[i]) - torch.mul(target[i], mask[i])))
                tmp/=torch.sum(mask[i])
                out+=tmp

            return out/len(input)



    def calc_dis_loss(self,outs0, outs1):
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        return loss


    def calc_gen_loss(self, outs0):
        # calculate the loss to train G
        loss = 0
        for it, (out0) in enumerate(outs0):
            loss += torch.mean((out0 - 1)**2) # LSGAN
        return loss




    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters, need_opt=True, path=None):
        # Load generators
        if (path == None):
            last_model_name = get_model_list(checkpoint_dir, "gen")
        else:
            last_model_name = path
        state_dict = torch.load(last_model_name)
        self.gen.module.load_state_dict(state_dict['gen'])
        if self.dis_scheduler is not None:
            self.dis.module.load_state_dict(state_dict['dis'])
        iterations = int(last_model_name[-11:-3])
        if (need_opt):
            self.gen_opt.load_state_dict(state_dict['gen_opt'])
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
            if self.dis_scheduler is not None:
                self.dis_opt.load_state_dict(state_dict['dis_opt'])
                self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        if self.dis_scheduler is not None:
            torch.save({'gen': self.gen.module.state_dict(), 'gen_opt': self.gen_opt.state_dict(),
                        'dis': self.dis.module.state_dict(), 'dis_opt': self.dis_opt.state_dict()}, gen_name)
        else:
            torch.save({'gen': self.gen.module.state_dict(), 'gen_opt': self.gen_opt.state_dict()}, gen_name)
