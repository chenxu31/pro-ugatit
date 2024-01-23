#-- coding:UTF-8 --
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .base_networks import get_norm_layer, ResnetBlock
except:
    from base_networks import get_norm_layer, ResnetBlock
import functools
from torch.nn.utils import spectral_norm

from math import sqrt

import torch.nn.functional as nnf

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.ReLU()
        )
        self.gamma = nn.Linear(dim, dim, bias=True)
        self.beta = nn.Linear(dim, dim, bias=True)
        self.dim = dim

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        x = x.view(b, c)
        x = self.mlp(x)
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta

class betaGamma_Down(nn.Module):
    def __init__(self, dim, out):
        super().__init__()

        self.gammaMLP = nn.Linear(dim, out, bias=True)
        self.betaMLP = nn.Linear(dim, out, bias=True)

        self.dim = dim
        self.out = out


    def forward(self, gamma, beta):
        gamma2 = self.gammaMLP(gamma)
        beta2 = self.betaMLP(beta)
        return gamma2, beta2


class AdaLIN(nn.Module):
    def __init__(self, anime):
        super().__init__()
        self.eps = 1e-6
        self.rho = nn.Parameter(torch.FloatTensor(1).fill_(0.9))
        self.anime = anime
    def forward(self, x, gamma, beta):
        b,c,h,w = x.shape
        ins_mean = x.view(b,c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_var = x.view(b,c,-1).var(dim=2) + self.eps
        ins_std = ins_var.sqrt().view(b,c ,1, 1)

        x_ins = (x - ins_mean) / ins_std

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        x_ln = (x - ln_mean) / ln_std
        if self.anime:
            rho = self.rho  # for sel->anime
        else:
            rho = (self.rho - 0.1).clamp(0, 1.0)  # smoothing, for adult->age
        x_hat = rho * x_ins + (1-rho) * x_ln
        x_hat = x_hat * gamma + beta ##Where cause the bug
        return x_hat

class ResBlockByAdaLIN(nn.Module):
    def __init__(self, dim, out_dim, anime=False):
        super().__init__()
        self.in_dim = dim
        self.out_dim = out_dim

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, out_dim, 3, 1, 0)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, out_dim, 3, 1, 0)
        )
        self.addin_1 = AdaLIN(anime)
        self.addin_2 = AdaLIN(anime)
        self.relu = nn.ReLU()

    def forward(self, x, gamma, beta):
        x1 = self.conv1(x)
        x1 = self.relu(self.addin_1(x1, gamma, beta))

        x2 = self.conv2(x1)
        x2 = self.addin_2(x2, gamma, beta)
        return x + x2



class ResBlock(nn.Module):
    def __init__(self, dim, out_dim, anime=False):
        super().__init__()
        self.in_dim = dim
        self.out_dim = out_dim

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, out_dim, 3, 1, 0)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_dim, out_dim, 3, 1, 0)
        )
        self.relu = nn.ReLU()

    def forward(self, x, gamma, beta):
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        return x1 + x2


class DecoderUpsamplingAdainModule(nn.Module):
    def __init__(self, dim, out_dim =256, anime=False,ifUpsampling=True):
        super().__init__()
        
        self.ifUpsampling = ifUpsampling
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.reflectionPadding2d = nn.ReflectionPad2d(1)             
        self.conv3x3 = nn.Conv2d(dim, out_dim, 3, 1, 0)
       
        
        self.res1 = ResBlock(dim, out_dim, anime=anime)
        #self.res2 = ResBlockByAdaLIN(out_dim, out_dim, anime=anime)

    def forward(self, x, gamma, beta):
        if(self.ifUpsampling):
            x1 = self.upsampling(x)
            #x1 = self.reflectionPadding2d(x1)
            #x1 = self.conv3x3(x1)
            x1 = self.res1(x1, gamma, beta)

            return x1
        
        else:
            x1 = x

        return x1

        x1 = self.res1(x1, gamma, beta)
        return x1
        #x1 = self.res2(x1, gamma, beta)
        #return x1


class LayerInstanceNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))
        self.rho = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        ins_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_val = x.view(b, c, -1).var(dim=2).view(b, c, 1, 1) + self.eps
        ins_std = ins_val.sqrt()

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        rho = torch.clamp(self.rho, 0, 1)
        x_ins = (x - ins_mean) / ins_std
        x_ln = (x - ln_mean) / ln_std

        x_hat = rho * x_ins + (1 - rho) * x_ln
        return x_hat * self.gamma + self.beta


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64,
                 use_dropout=False, n_blocks=4,
                 padding_type='reflect', anime=False):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.n_blocks = n_blocks
        instance_norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
        model = []

        mult = 1/4
        downsampling4x = [ nn.Conv2d(int(ngf * mult), int(ngf * mult*2), kernel_size=3, stride=2, padding=1, bias=True),
                           nn.InstanceNorm2d(int(ngf * mult*2), affine=True),
                           nn.ReLU(True)]
        mult = 1/2
        downsampling2x = [ nn.Conv2d(int(ngf * mult), int(ngf * mult*2), kernel_size=3, stride=2, padding=1, bias=True),
                           nn.InstanceNorm2d(int(ngf * mult*2), affine=True),
                           nn.ReLU(True)]
        mult = 1
        downsampling1x = [ nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                           nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                           nn.ReLU(True)]
        mult = 2
        downsampling_s2 = [ nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                           nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                           nn.ReLU(True)]
        #mult = 4
        #downsampling_s4 = [ nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, bias=True),
                           #nn.InstanceNorm2d(ngf * mult, affine=True),
                           #nn.ReLU(True)]
        mult = 0.25
        from_4x = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, int(ngf*mult), kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]
        mult = 0.5
        from_2x = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, int(ngf*mult), kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]
        mult = 1
        from_1x = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, int(ngf*mult), kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]
        mult = 2
        from_s2 = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]
        mult = 4
        from_s4 = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]

       
       
        Ldownsampling4x = nn.Sequential(*downsampling4x)
        Ldownsampling2x = nn.Sequential(*downsampling2x)
        Ldownsampling1x = nn.Sequential(*downsampling1x)
        Ldownsampling_s2 = nn.Sequential(*downsampling_s2)


        self.DWM= nn.ModuleList([Ldownsampling_s2,Ldownsampling1x,Ldownsampling2x,Ldownsampling4x])
        

        Lfrom_4x = nn.Sequential(*from_4x)
        Lfrom_2x = nn.Sequential(*from_2x)
        Lfrom_1x = nn.Sequential(*from_1x)
        Lfrom_s2 = nn.Sequential(*from_s2)
        Lfrom_s4 = nn.Sequential(*from_s4)
        
        self.fromRGBs = nn.ModuleList([Lfrom_s4,Lfrom_s2,Lfrom_1x,Lfrom_2x,Lfrom_4x])



                      
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=instance_norm_layer, use_dropout=use_dropout,
                                  use_bias=True)]
        self.encoder = nn.Sequential(*model)

        # CAM
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.cam_w = nn.Parameter(torch.FloatTensor(ngf*mult, 1))
        nn.init.xavier_uniform_(self.cam_w)
        self.cam_bias = nn.Parameter(torch.FloatTensor(1))
        self.cam_bias.data.fill_(0)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2*ngf*mult, ngf*mult, 1, 1),
            nn.ReLU(),
        )
        # MLP
        self.mlp = MLP(ngf*mult)

        adain_resblock = []
        for i in range(n_blocks):
            adain_resblock.append(ResBlockByAdaLIN(ngf*mult,ngf*mult,anime))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        decoder_s4 =[]
        decoder_s2 =[]
        decoder =[]
        decoder2x =[]
        decoder4x = []
        
        
        decoder_s4 = [DecoderUpsamplingAdainModule(ngf*mult, ngf*mult, anime=anime, ifUpsampling=False)]
        
        self.decoder_s4 = nn.ModuleList(decoder_s4)

      
        decoder_s2 = [DecoderUpsamplingAdainModule(ngf*mult, ngf*mult, anime=anime, ifUpsampling=True)]
        self.decoder_s2 = nn.ModuleList(decoder_s2)
      
                      

        
        
        decoder = [DecoderUpsamplingAdainModule(ngf*mult, ngf*mult, anime=anime, ifUpsampling=True),
                   betaGamma_Down(ngf*mult, ngf*mult//2)]
        self.decoder = nn.ModuleList(decoder)
      
        decoder2x = [DecoderUpsamplingAdainModule(ngf*mult, ngf*mult//2, anime=anime, ifUpsampling=True),
                     betaGamma_Down(ngf*mult//2, ngf*mult//4)]
        self.decoder2x = nn.ModuleList(decoder2x)

        decoder4x = [DecoderUpsamplingAdainModule(ngf*mult//2, ngf*mult//4, anime=anime, ifUpsampling=True),
                    betaGamma_Down(ngf*mult//4, ngf*mult//8)]
        self.decoder4x = nn.ModuleList(decoder4x)

        
        output_s4 =[nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf*mult, output_nc, 7, 1),
                       nn.Tanh()]
        output_s2 =[nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf*mult, output_nc, 7, 1),
                       nn.Tanh()]
        output = [nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf*mult, output_nc, 7, 1),
                       nn.Tanh()]
        output2x = [nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf*mult//2, output_nc, 7, 1),
                       nn.Tanh()]
        output4x = [nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf*mult//4, output_nc, 7, 1),
                       nn.Tanh()]

        self.output_s4 = nn.Sequential(*output_s4)
        self.output_s2 = nn.Sequential(*output_s2)
        self.output = nn.Sequential(*output)
        self.output2x = nn.Sequential(*output2x)
        self.output4x = nn.Sequential(*output4x)   

        self.max_step = 4
        



    def cam(self, e):
        b, c, h, w = e.shape
        gap = self.gap(e).view(b, c)
        gmp = self.gmp(e).view(b, c)

        x_a = torch.matmul(gap, self.cam_w) + self.cam_bias  # for classfication loss
        x_m = torch.matmul(gmp, self.cam_w) + self.cam_bias

        x_gap = e * (self.cam_w + self.cam_bias).view(1, c, 1, 1)
        x_gmp = e * (self.cam_w + self.cam_bias).view(1, c, 1, 1)

        x = torch.cat((x_gap, x_gmp), dim=1)
        x = self.conv1x1(x)
        x_class = torch.cat((x_a, x_m), dim=1) # b, 2
        return x, x_class

    def forward(self, img, step=0,alpha=0):
        
        e = self.fromRGBs[step](img)

        for i in range(step-1, -1, -1):
            e = self.DWM[i](e)
  
        e = self.encoder(e)
        x, x_class = self.cam(e)
        b, c, h, w = x.shape
        gamma, beta = self.mlp(self.gap(x).view(b, c))  # predict beta, gamma of  adain

        og = gamma
        ob = beta

        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)

      
        

        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, gamma, beta)

        x = self.decoder_s4[0](x, gamma, beta)
        if step==0:
            x = self.output_s4(x)
            return x, x_class
        
        x = self.decoder_s2[0](x, gamma, beta)
        if step==1:
            x = self.output_s2(x)
            return x, x_class
        
        x = self.decoder[0](x, gamma, beta)
        #og, ob = self.decoder[1](og, ob)
        #gamma = og.view(b, int(c/2), 1, 1)
        #beta = ob.view(b, int(c/2), 1, 1)

        if step==2:
            x = self.output(x)
            return x,x_class

        x = self.decoder2x[0](x, gamma, beta)
        #og, ob = self.decoder2x[1](og, ob)
        #gamma = og.view(b, int(c/4), 1, 1)
        #beta = ob.view(b, int(c/4), 1, 1)

        if step==3:
            x = self.output2x(x)
            return x, x_class
        
        x = self.decoder4x[0](x, gamma, beta)
        #gamma,beta = self.decoder4x[1](gamma, beta) #when 8x available
        if step==4:
            x = self.output4x(x)
            return x, x_class
        
        
        
        
        return x, x_class

    def test_forward(self, x, dir=None,step=0):
        return self.forward(x,step=step)[0]

    def upscale(feat):
        return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)


class CAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.weight = nn.Parameter(torch.FloatTensor(dim, 1))
        nn.init.xavier_uniform_(self.weight)
        self.cam_bias = nn.Parameter(torch.FloatTensor(1))
        self.cam_bias.data.fill_(0)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, 1),
            nn.LeakyReLU(0.2),
        )
    def forward(self, e):
        b, c, h, w = e.shape
        gap = self.gap(e).view(b, c)
        gmp = self.gmp(e).view(b, c)

        x_a = torch.matmul(gap, self.weight) + self.cam_bias  # for classfication loss
        x_m = torch.matmul(gmp, self.weight) + self.cam_bias

        x_gap = e * (self.weight + self.cam_bias).view(1, c, 1, 1)
        x_gmp = e * (self.weight + self.cam_bias).view(1, c, 1, 1)

        x = torch.cat((x_gap, x_gmp), dim=1)
        x = self.conv1x1(x)
        x_class = torch.cat((x_a, x_m), dim=1)  # b, 2
        return x, x_class

class Discriminator(nn.Module):
    def __init__(self, inc, ndf, n_blocks=6):
        super().__init__()
        # local


        from_s4 = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf*2, 4, 1)),  #input: (h/4,w/4,3) -> (h/4,h/4,128)
            nn.LeakyReLU(0.2),
        ]
        from_s2 = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf, 4, 1)),  #input: (h/2,w/2,3) -> (h/2,h/2,64)
            nn.LeakyReLU(0.2),
        ]
        
        from1x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf, 4, 2)),  #input: (h,w,3) -> (h/2,h/2,64)
            nn.LeakyReLU(0.2),
        ]


        from2x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf//2, 4, 2)),  #input: (2h,2w,3) -> (h,w,32)
            nn.LeakyReLU(0.2)
        ]
        from4x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf//4, 4, 2)),  #input: (4h,4w,3) -> (2h,2w,16)
            nn.LeakyReLU(0.2)
        ]

        ##############################

        #(2h,2w,16) => (h,w,32)
        _2xTo1x_ =[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf//4, ndf//2, 4, 2)),#input: (2h,2w,16) -> (h,w,32)
            nn.LeakyReLU(0.2)
        ]

        #(h,w,32) => (h/2,w/2,64)
        
        _1xToDot5x =[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf//2, ndf, 4, 2)),#input: (h,w,32) -> (h/2,w/2,64)
            nn.LeakyReLU(0.2)
        ]

        #(h/2,w/2,64) => (h/4,w/4,128)
     
        _Dot5xToDot25x_=[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2)),#input: (h/2,w/2,64) -> (h/4,w/4,128)
            nn.LeakyReLU(0.2)
        ]


        #Unbounded Area
        #input: (h/4,w,4,128) -> (h/8,h/8,256)

        mult = 2
        local = [
            
            nn.ReflectionPad2d(1),                            #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf*2, ndf*mult*2, 4, 2)),  #input: (h/4,w,4,128) -> (h/8,h/8,256)
            nn.LeakyReLU(0.2),
        ]

        #n_block = 6 , range= (1,2)
        #(h/8,w/8,256) => (h/8,w/8,512)
        for i in range(1, n_blocks-2-1-1):
            mult = mult * 2 
            local.extend([
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(ndf*mult, ndf*mult*2, 4, 1)),
                nn.LeakyReLU(0.2),
            ])

        '''mult = mult * 2
        
        local.extend([
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf*mult, ndf*mult*2, 4, 1)),
            nn.LeakyReLU(0.2)
        ])'''


        ##add to sequential 
        self.from_s4 = nn.Sequential(*from_s4)
        self.from_s2 = nn.Sequential(*from_s2)
        self.from1x = nn.Sequential(*from1x)
        self.from2x = nn.Sequential(*from2x)
        self.from4x = nn.Sequential(*from4x)

        self._2xTo1x_ = nn.Sequential(*_2xTo1x_)
        self._1xToDot5x = nn.Sequential(*_1xToDot5x)
        self._Dot5xToDot25x_ = nn.Sequential(*_Dot5xToDot25x_)


        self.local_base = nn.Sequential(*local)
        mult = mult * 2
        self.local_cam = spectral_norm(CAM(mult*ndf))
        self.local_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(mult*ndf, 1, 4, 1)),
        )




        # global #################################################################

        global_from_s4 = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf*2, 4, 1)),  #input: (h/4,w/4,3) -> (h/4,h/4,128)
            nn.LeakyReLU(0.2),
        ]
        global_from_s2 = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf, 4, 1)),  #input: (h/2,w/2,3) -> (h/2,h/2,64)
            nn.LeakyReLU(0.2),
        ]
        
        global_from1x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf, 4, 2)),  #input: (h,w,3) -> (h/2,h/2,64)
            nn.LeakyReLU(0.2),
        ]


        global_from2x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf//2, 4, 2)),  #input: (2h,2w,3) -> (h,w,32)
            nn.LeakyReLU(0.2)
        ]
        global_from4x = [
            nn.ReflectionPad2d(1),                     #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(inc, ndf//4, 4, 2)),  #input: (4h,4w,3) -> (2h,2w,16)
            nn.LeakyReLU(0.2)
        ]

        ##############################

        #(2h,2w,16) => (h,w,32)
        global_2xTo1x_ =[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf//4, ndf//2, 4, 2)),#input: (2h,2w,16) -> (h,w,32)
            nn.LeakyReLU(0.2)
        ]

        #(h,w,32) => (h/2,w/2,64)
        
        global_1xToDot5x =[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf//2, ndf, 4, 2)),#input: (h,w,32) -> (h/2,w/2,64)
            nn.LeakyReLU(0.2)
        ]

        #(h/2,w/2,64) => (h/4,w/4,128)
     
        global_Dot5xToDot25x_=[
            nn.ReflectionPad2d(1),                         #ndf = 64 (orignalSize)
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2)),#input: (h/2,w/2,64) -> (h/4,w/4,128)
            nn.LeakyReLU(0.2)
        ]


        #Unbounded Area
        #input: (h/4,w,4,128) -> (h/8,h/8,256)

        mult = 2
        global_ = [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf*2, ndf*mult*2, 4, 2)), #input: (h/4,w,4,128) -> (h/8,h/8,256)
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, n_blocks -1 -2 -1):
            mult = 2* mult
            global_.extend([
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2)), #256->512->1024->2048
                nn.LeakyReLU(0.2),
            ])
        mult *= 2
        global_.extend([
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 1)),
            nn.LeakyReLU(0.2)
        ])
        mult *= 2


        self.global_from_s4 = nn.Sequential(*global_from_s4)
        self.global_from_s2 = nn.Sequential(*global_from_s2)
        self.global_from1x = nn.Sequential(*global_from1x)
        self.global_from2x = nn.Sequential(*global_from2x)
        self.global_from4x = nn.Sequential(*global_from4x)

        self.global_2xTo1x_ = nn.Sequential(*global_2xTo1x_)
        self.global_1xToDot5x = nn.Sequential(*global_1xToDot5x)
        self.global_Dot5xToDot25x_ = nn.Sequential(*global_Dot5xToDot25x_)



        self.global_base = nn.Sequential(*global_)
        self.global_cam = spectral_norm(CAM(mult*ndf))
        self.global_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(mult * ndf, 1, 4, 1)),
        )

    def forward(self, img, step=0, alpha=0):
        if(step==0):
            x = self.from_s4(img)
        if(step==1):
            x = self.from_s2(img) 
            x = self._Dot5xToDot25x_(x)
            
        if(step==2):
            x = self.from1x(img)
            x = self._Dot5xToDot25x_(x)
        if(step==3):
            x = self.from2x(img)
            x = self._1xToDot5x(x)
            x = self._Dot5xToDot25x_(x)
        if(step==4):
            x = self.from4x(img)
            x = self._2xTo1x_(x)
            x = self._1xToDot5x(x)
            x = self._Dot5xToDot25x_(x)


        local_base = self.local_base(x)
        local_x, local_class = self.local_cam(local_base)
        local_x = self.local_head(local_x)



        if(step==0):
            x = self.global_from_s4(img)
        if(step==1):
            x = self.global_from_s2(img) 
            x = self.global_Dot5xToDot25x_(x)
            
        if(step==2):
            x = self.global_from1x(img)
            x = self.global_Dot5xToDot25x_(x)
        if(step==3):
            x = self.global_from2x(img)
            x = self.global_1xToDot5x(x)
            x = self.global_Dot5xToDot25x_(x)
        if(step==4):
            x = self.global_from4x(img)
            x = self.global_2xTo1x_(x)
            x = self.global_1xToDot5x(x)
            x = self.global_Dot5xToDot25x_(x)

        global_base = self.global_base(x)
        global_x, global_class = self.global_cam(global_base)
        global_x = self.global_head(global_x)
        # print(local_x.shape, global_x.shape, global_class.shape)
        return local_x, local_class, global_x, global_class



class UGATIT(object):
    def __init__(self, args):
        super().__init__()
        self.G_A = Generator(args.inc, args.outc, args.ngf, args.use_dropout, args.n_blocks, anime=args.anime)
        self.G_B = Generator(args.inc, args.outc, args.ngf, args.use_dropout, args.n_blocks, anime=args.anime)
        if args.training:
            self.D_A = Discriminator(args.inc, args.ndf, args.d_layers)
            self.D_B = Discriminator(args.inc, args.ndf, args.d_layers)
        self.training = args.training
        self.size = args.load_size

        #self.blurModule = GaussianSmoothing(3, 9, 24)
        self.L1Loss = nn.L1Loss()


    def __call__(self, inp, step=0, alpha=0):
        realA, realB = inp['A'], inp['B']
        if torch.cuda.is_available():
            realA, realB = realA.cuda(), realB.cuda()
        fakeB, cam_ab = self.G_A(realA,step=step)
        fakeA, cam_ba = self.G_B(realB,step=step)

        #fakeAIn = nnf.interpolate(fakeA, size=self.size, mode='bilinear',align_corners=False)
        #fakeBIn = nnf.interpolate(fakeB, size=self.size, mode='bilinear',align_corners=False)

        recA, _ = self.G_B(fakeB,step=step)
        recB, _ = self.G_A(fakeA,step=step)

        return realA, realB, fakeA, fakeB, recA, recB, cam_ab, cam_ba

    def train(self):
        self.D_A.train()
        self.G_A.train()
        self.D_B.train()
        self.G_B.train()

    def cuda(self):
        self.G_A.cuda()
        self.G_B.cuda()
        if self.training:
            self.D_A.cuda()
            self.D_B.cuda()
            
       
        self.L1Loss.cuda()

    def test_forward(self, x, dir):
        raise NotImplementedError


    def state_dict(self):
        params = {
            'G_A': self.G_A.state_dict(), # dist training
            'G_B': self.G_B.state_dict(),
        }
        if self.training:
            params['D_A'] = self.D_A.state_dict(),
            params['D_B'] = self.D_B.state_dict()
        return params

    def load_state_dict(self, weight_loc):
        weight_set = torch.load(weight_loc, map_location='cpu')
        self.G_A.load_state_dict(weight_set['G_A'])
        self.G_B.load_state_dict(weight_set['G_B'])
        if self.training:
            self.D_A.load_state_dict(weight_set['D_A'])
            self.D_B.load_state_dict(weight_set['D_B'])


if __name__ == '__main__':
    
    # model = UGATIT(3, 3, 64, True, n_blocks=4)  # 52G
    # model.eval()
    pass








