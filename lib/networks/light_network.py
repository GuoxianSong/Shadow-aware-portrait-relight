
from torch import nn
from lib.networks.base_network import DynamicMlp,Conv2dBlock,ResBlocks,LinearBlock,MsImageDis
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
    def __init__(self, input_dim, params):
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
        model = torch.hub.load('pytorch/vision:v0.3.0', 'densenet121', pretrained=True)
        self.model = model.features[:-1]
        self.pool  = torch.nn.AvgPool2d(5)
        self.decoder = LinearBlock(1024*2,32*16*3,activation='tanh')


    def forward(self, images):
        # reconstruct an image
        features = self.pool(self.model(images))
        out = self.decoder(features.view(features.size(0),1024*2))
        return out.view(out.size(0),3,16,32)



