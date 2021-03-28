
from torch import nn
from lib.networks.base_network import DynamicMlp,Conv2dBlock,ResBlocks,LinearBlock
import torch
from torch.autograd import Variable
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

##################################################################################
# Generator
##################################################################################

class Gen(nn.Module):
    # auto-encoder architecture
    def __init__(self, input_dim, params,live=False):
        super(Gen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm_type = params['norm_type']


        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ,
                                              pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm=norm_type, activ=activ,
                           pad_type=pad_type,live=live)

        self.dec_p = Decoder(n_downsample, n_res, self.enc_content.output_dim, 1, res_norm=norm_type, activ=activ,
                           pad_type=pad_type,unet=False)

    def forward(self, images,require_depth = True):
        # reconstruct an image
        x, x_embed = self.enc_content(images)
        images_recon = self.dec(x, x_embed)
        if(require_depth):
            p_recon = self.dec_p(x, x_embed)
            return images_recon,p_recon
        else:
            return images_recon

    # def live_forward(self,images):
    #     x, x_embed = self.enc_content(images)
    #     images_recon = self.dec(x, x_embed)
    #     return images_recon
##################################################################################
# Encoder and Decoders
##################################################################################



class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample-1):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        out=[]
        for i in range(len(self.model)):
            out.append(x)
            x = self.model[i](x)
        return x,out

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='my_render', activ='relu', pad_type='zero',unet=True,live =False):
        super(Decoder, self).__init__()
        self.unet = unet
        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        if(unet):
            for i in range(n_upsample):
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(dim*2, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
                dim //= 2
            # use reflection padding in the last conv layer
            if(live):
                self.model += [Conv2dBlock(dim*3+3, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='zero')]
            else:
                self.model += [
                    Conv2dBlock(dim * 3 + 3, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        else:
            for i in range(n_upsample):
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
                dim //= 2
            # use reflection padding in the last conv layer
            self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]


        self.model = nn.Sequential(*self.model)

    def forward(self, x,embed):
        x = self.model[0](x)
        if(self.unet):
            x = torch.cat((x,embed[3]),dim=1)
        x = self.model[1](x)
        x = self.model[2](x)
        if(self.unet):
            x = torch.cat((x,embed[2]),dim=1)
        x = self.model[3](x)
        x = self.model[4](x)
        if(self.unet):
            x = torch.cat((x, embed[1],embed[0]), dim=1)
        x = self.model[5](x)
        return x




