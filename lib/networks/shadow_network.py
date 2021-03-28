
from torch import nn
from lib.networks.base_network import DynamicMlp,Conv2dBlock,ResBlocks,LinearBlock,MLP
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
    def __init__(self, params):
        super(Gen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        norm_type = params['norm_type']
        input_dim_gen = params['input_dim_gen']
        output_dim_gen = params['output_dim_gen']

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim_gen, dim, 'in', activ,
                                              pad_type=pad_type)

        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_dim_gen,dim, res_norm=norm_type, activ=activ,
                           pad_type=pad_type)

        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none',
                       activ=activ)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def forward(self, depth,light):
        # reconstruct an image
        x = self.enc_content(depth)

        adain_params = self.mlp(light)
        self.assign_adain_params(adain_params, self.dec)

        images_recon = self.dec(x)
        return images_recon




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
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [Conv2dBlock(dim, dim,3, 1, 1, norm=norm, activation=activ, pad_type=pad_type,atten_conv=True)]
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, gen_dim, res_norm='my_render', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample+1):
            if(i==0):
                self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(int(dim), dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            else:
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(int(dim ), dim // 2, 5, 1, 2, norm='ln', activation=activ,
                                           pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)



    def forward(self, x):
        x = self.model(x)
        return x






