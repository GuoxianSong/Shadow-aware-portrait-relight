
from torch import nn
from lib.networks.base_network import DynamicMlp,Conv2dBlock,ResBlocks,LinearBlock,MLP,DecoderBlockV2,ConvRelu
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
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



        self.enc = MyEncoder()
        self.dec = MyDecoder(n_downsample, n_res, self.enc.output_dim, output_dim_gen,dim, res_norm=norm_type, activ=activ,
                             pad_type=pad_type)

        self.mlp = MLP(style_dim, self.get_num_params(self.dec), mlp_dim, 3, norm='none',
                                      activ=activ)



    def assign_params(self, _params, model,middle =None):
        # assign the mul/add_params to the render layers in model
        for m in model.modules():
            if m.__class__.__name__ == "MyRenderLayer":
                if(middle!=None):
                    mean = middle[:, :m.num_features]
                else:
                    mean = _params[:, :m.num_features]
                std = _params[:, m.num_features:2 * m.num_features]
                m.bias = mean
                m.weight = std
                if _params.size(1) > 2 * m.num_features:
                    _params = _params[:, 2 * m.num_features:]

    def get_num_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "MyRenderLayer":
                num_params += 2 * m.num_features
        return num_params

    def forward(self, images,shadow_image,light):
        # reconstruct an image
        x, x_embed = self.enc(images)
        light_code = self.mlp(light)
        self.assign_params(light_code, self.dec)
        images_recon = self.dec(x, x_embed,shadow_image)
        return images_recon
##################################################################################
# Encoder and Decoders
##################################################################################


class MyEncoder(nn.Module):
    def __init__(self):
        super(MyEncoder, self).__init__()
        self.conv_down1_1 = Conv2dBlock(3, 29, 7,padding=3)
        self.conv_down1_2 = Conv2dBlock(32, 64, stride=2,padding=1)
        self.conv_down2_1 = Conv2dBlock(64, 64,padding=1)
        self.conv_down2_2 = Conv2dBlock(64, 128, stride=2,padding=1)
        self.conv_down3_1 = Conv2dBlock(128, 128,padding=1)
        self.conv_down3_2 = Conv2dBlock(128, 256, stride=2,padding=1)
        self.conv_down4_1 = Conv2dBlock(256, 256,padding=1)
        self.conv_down4_2 = Conv2dBlock(256, 512, stride=2,padding=1)

        self.conv_down5_1 = Conv2dBlock(512, 512,padding=1)
        self.conv_down5_2 = Conv2dBlock(512, 512,padding=1)
        self.conv_down5_3 = Conv2dBlock(512, 512,padding=1)
        self.conv_down5_4 = Conv2dBlock(512, 512, stride=2,padding=1, activation='softplus')
        self.output_dim = 512

    def forward(self, x):
        conv1_1 = torch.cat((self.conv_down1_1(x), x), dim=1)
        conv1_2 = self.conv_down1_2(conv1_1)

        conv2_1 = self.conv_down2_1(conv1_2)
        conv2_2 = self.conv_down2_2(conv2_1)

        conv3_1 = self.conv_down3_1(conv2_2)
        conv3_2 = self.conv_down3_2(conv3_1)

        conv4_1 = self.conv_down4_1(conv3_2)
        conv4_2 = self.conv_down4_2(conv4_1)

        conv5_1 = self.conv_down5_1(conv4_2)
        conv5_2 = self.conv_down5_2(conv5_1)
        conv5_3 = self.conv_down5_3(conv5_2)
        conv5_4 = self.conv_down5_4(conv5_3)

        embed = [conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv4_1, conv4_2,
                 conv5_1, conv5_2]
        return conv5_4, embed

class MyDecoder(nn.Module):
    def __init__(self,n_upsample, n_res, dim, output_dim, gen_dim, res_norm='my_render', activ='relu', pad_type='zero'):
        super(MyDecoder, self).__init__()

        self.relight = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        self.conv_shadow = Conv2dBlock(1, 31, 7, padding=3)

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_up5_3 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(512,256,padding=1)])
        self.conv_up5_2 = Conv2dBlock(512+256, 512,padding=1)
        self.conv_up5_1 = Conv2dBlock(512+512, 512,padding=1)

        self.conv_up4_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(512+512, 256,padding=1)])
        self.conv_up4_1 = Conv2dBlock(256+256, 256,padding=1)

        self.conv_up3_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(256+256, 128,padding=1)])
        self.conv_up3_1 = Conv2dBlock(128+128, 128,padding=1)

        self.conv_up2_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(128+128, 64,padding=1)])
        self.conv_up2_1 = Conv2dBlock(64+64, 64,padding=1)

        self.conv_up1_2 = nn.Sequential(*[nn.Upsample(scale_factor=2),Conv2dBlock(64 + 64, 32,padding=1)])
        self.conv_up1_1 = Conv2dBlock(32 + 32, 3,padding=1,activation='tanh')



    def forward(self, x,embed,shadow):  # b x 3 x 16 x 32
        x = self.relight(x)
        shadow_embed =1-torch.cat((self.conv_shadow(shadow), shadow), dim=1)
        # tilling
        dconv_up5_3 = torch.cat((embed[9], self.conv_up5_3(x)), dim=1)
        dconv_up5_2 = torch.cat((embed[8], self.conv_up5_2(dconv_up5_3)), dim=1)
        dconv_up5_1 = torch.cat((embed[7], self.conv_up5_1(dconv_up5_2)), dim=1)

        dconv_up4_2 = torch.cat((embed[6], self.conv_up4_2(dconv_up5_1)), dim=1)
        dconv_up4_1 = torch.cat((embed[5], self.conv_up4_1(dconv_up4_2)), dim=1)

        dconv_up3_2 = torch.cat((embed[4], self.conv_up3_2(dconv_up4_1)), dim=1)
        dconv_up3_1 = torch.cat((embed[3], self.conv_up3_1(dconv_up3_2)), dim=1)

        dconv_up2_2 = torch.cat((embed[2], self.conv_up2_2(dconv_up3_1)), dim=1)
        dconv_up2_1 = torch.cat((embed[1], self.conv_up2_1(dconv_up2_2)), dim=1)



        dconv_up1_2 = torch.cat((torch.mul(embed[0],shadow_embed), torch.mul(self.conv_up1_2(dconv_up2_1),shadow_embed)), dim=1)
        dconv_up1_1 = self.conv_up1_1(dconv_up1_2)

        return dconv_up1_1