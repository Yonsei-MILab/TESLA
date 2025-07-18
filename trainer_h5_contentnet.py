"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks_contentnet import AdaINGen, PatchGAN_Dis
from utils import weights_init, get_model_list, vgg_preprocess, get_scheduler, get_model_ckpt_name
import torch
import torch.nn as nn
import os
from perceptual import PerceptualLoss
from monai.losses.ssim_loss import SSIMLoss
import numpy as np

torch.manual_seed(0)

class ContentNet_Trainer(nn.Module):
    def __init__(self, config):
        super(ContentNet_Trainer, self).__init__()
        
        # setup the usable gpu
        self.device = torch.device("cuda:" + config.device)

        # Data options
        self.input_ch_a = config.input_ch_a
        self.input_ch_b = config.input_ch_b
        
        # Model configurations
        # Generator
        self.gen_dim          = config.gen_dim
        self.gen_mlp_dim      = config.gen_mlp_dim
        self.gen_style_dim    = config.gen_style_dim
        self.gen_activ        = config.gen_activ
        self.gen_n_downsample = config.gen_n_downsample
        self.gen_n_res        = config.gen_n_res
        self.gen_pad_type     = config.gen_pad_type

        # Discriminator
        self.dis_dim        = config.dis_dim       
        self.dis_norm       = config.dis_norm      
        self.dis_activ      = config.dis_activ     
        self.dis_n_layer    = config.dis_n_layer   
        self.dis_gan_type   = config.dis_gan_type  
        self.dis_num_scales = config.dis_num_scales
        self.dis_pad_type   = config.dis_pad_type

        # Train configurations
        self.dataset = config.dataset
        self.workers = config.workers
        self.epochs  = config.epochs
        self.batch_size = config.batch_size
        self.gen_lr = config.gen_lr
        self.dis_lr = config.dis_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # self.gpu = gpu

        # Test configurations.
        self.test_epochs = config.test_epochs

        # Initiate the networks
        self.gen_a = AdaINGen(self.input_ch_a, config)  # auto-encoder for domain a
        self.dis_a = PatchGAN_Dis() # discriminator for domain a
        
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = config.gen_style_dim
        self.conv2d_1x1 = nn.Conv2d(in_channels=config.gen_dim, out_channels=1, kernel_size=1)

        # loss
        self.lsgan = True
        self.ganloss  = torch.nn.MSELoss()
        self.L1loss   = torch.nn.L1Loss()
        self.SSIMloss = SSIMLoss(spatial_dims=2, data_range=1)

        # fix the noise used in sampling
        batch_size = int(config.batch_size)                                              
        self.s_a   = torch.randn(batch_size, self.style_dim, 1, 1).to( self.device )

        # Setup the optimizers
        beta1 = config.beta1
        beta2 = config.beta2

        dis_params = list(self.dis_a.parameters())
        gen_params = list(self.gen_a.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=self.dis_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=self.gen_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.gen_scheduler = get_scheduler(self.gen_opt, config)

        # Network weight initialization
        self.apply(weights_init(config.init))
        self.dis_a.apply(weights_init('gaussian'))

        # Miscellaneous
        self.dis_loss_tb = {}
        self.gen_loss_tb = {}

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def dis_update(self, x_a, config, *dwt_canny):
        
        canny_edge  = dwt_canny[-1].float()

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, x_a.shape[-2] // 2 ** 4, x_a.shape[-1] // 2 ** 4)

        # Adversarial ground truths
        real = torch.Tensor(np.ones((x_a.size(0), *patch))).to(self.device)
        fake = torch.Tensor(np.zeros((x_a.size(0), *patch))).to(self.device)

        self.dis_opt.zero_grad()
        
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        
        # decode
        x_a_recon = self.gen_a.decode(c_a[-1], s_a_prime)

        # D Real loss
        pred_real = self.dis_a( img           = x_a[:],     
                                img_condition = canny_edge )
        loss_real = self.ganloss( pred_real, real )

        # D Fake loss
        pred_fake = self.dis_a( img           = x_a_recon.detach(), # (B, 1, 128, 256)
                                img_condition = canny_edge )        # (B, 1, 128, 256)
        loss_fake = self.ganloss( pred_fake, fake )
        
        # D Total loss
        self.loss_dis_a_total = 0.5 * ( loss_real + loss_fake )        
        self.loss_dis_total = config.gan_w * (self.loss_dis_a_total)
        
        # D loss for tensorboard visualization
        self.dis_loss_tb['D_loss/dis_adv_a'] = self.loss_dis_total

        self.loss_dis_total.backward()
        self.dis_opt.step()
        
        return self.loss_dis_total, self.dis_loss_tb, self.dis_opt, pred_real, pred_fake
    
    def gen_update(self, x_a, config, data_A_edge ):

        canny_edge  = data_A_edge.float()

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, x_a.shape[-2] // 2 ** 4, x_a.shape[-1] // 2 ** 4)

        # Adversarial ground truths
        real = torch.Tensor(np.ones((x_a.size(0), *patch))).to(self.device)
        fake = torch.Tensor(np.zeros((x_a.size(0), *patch))).to(self.device)

        self.gen_opt.zero_grad()
        
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).to( self.device ) 
        
        
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        
        # decode
        x_a_recon = self.gen_a.decode(c_a[-1], s_a_prime)

        # encode again
        c_a_recon, s_a_recon = self.gen_a.encode(x_a_recon)
        
        # decode again (if needed)
        x_a_recon_a = self.gen_a.decode(c_a_recon[-1], s_a_recon) if config.recon_x_cyc_w > 0 else None


        ################################################
        ##################### Loss #####################
        ################################################
        # G Fake loss
        pred_fake = self.dis_a( img           = x_a_recon,
                                img_condition = canny_edge )
        loss_fake = self.ganloss( pred_fake, real )
        
        # G Total loss
        self.loss_gen_a_total = loss_fake
        self.loss_gen_adv_total = config.gan_w * (self.loss_gen_a_total)

        # pixel-wise reconstruction loss
        # L1 loss
        self.l1loss_gen_recon_x_a    = config.recon_l1_w * self.L1loss(x_a_recon, x_a[:]) # within-domain recon (t1)
        self.l1loss_gen_recon_s_a    = config.recon_l1_w * self.L1loss(s_a_recon, s_a_prime) # within-domain style recon (t1)
        self.l1loss_gen_recon_c_a    = config.recon_l1_w * self.L1loss(c_a_recon[-1], c_a[-1]) # cross-domain content recon (t1)
        self.l1loss_gen_cycrecon_x_a = config.recon_l1_w * self.L1loss(x_a_recon_a, x_a[:]) if config.recon_x_cyc_w > 0 else 0
        
        # SSIM loss
        self.ssimloss_gen_recon_x_a    = config.recon_ssim_w * self.SSIMloss(x_a_recon, x_a[:])
        self.ssimloss_gen_recon_c_a    = config.recon_ssim_w * self.SSIMloss(c_a_recon[-1], c_a[-1])
        self.ssimloss_gen_cycrecon_x_a = config.recon_ssim_w * self.SSIMloss(x_a_recon_a, x_a[:]) if config.recon_x_cyc_w > 0 else 0
        
        # reconstruction loss for tensorboard visualization
        # L1 loss
        self.gen_loss_tb['G_loss/L1_recon_x_a']         = self.l1loss_gen_recon_x_a   
        self.gen_loss_tb['G_loss/L1_recon_s_a']         = self.l1loss_gen_recon_s_a   
        self.gen_loss_tb['G_loss/L1_recon_c_a']         = self.l1loss_gen_recon_c_a   
        self.gen_loss_tb['G_loss/L1_recon_cyc_x_a']     = self.l1loss_gen_cycrecon_x_a
        
        # SSIM loss
        self.gen_loss_tb['G_loss/SSIM_recon_x_a']         = self.ssimloss_gen_recon_x_a   
        self.gen_loss_tb['G_loss/SSIM_recon_c_a']         = self.ssimloss_gen_recon_c_a   
        self.gen_loss_tb['G_loss/SSIM_recon_cyc_x_a']     = self.ssimloss_gen_cycrecon_x_a
        
        # gan loss for tensorboard viusalization
        self.gen_loss_tb['G_loss/gan_adv_a'] = self.loss_gen_adv_total
        
        # total loss
        self.loss_gen_total = self.l1loss_gen_recon_x_a      + \
                              self.l1loss_gen_recon_s_a      + \
                              self.l1loss_gen_recon_c_a      + \
                              self.l1loss_gen_cycrecon_x_a   + \
                              self.ssimloss_gen_recon_x_a    + \
                              self.ssimloss_gen_recon_c_a    + \
                              self.ssimloss_gen_cycrecon_x_a + \
                              self.loss_gen_adv_total

        self.loss_gen_total.backward()
        self.gen_opt.step()

        return self.loss_gen_total, self.gen_loss_tb, self.gen_opt, c_a, x_a_recon

    def update_learning_rate(self, config, val_loss=None):
        if self.dis_scheduler is not None:
            if config.lr_policy == 'step':
                self.dis_scheduler.step()
            elif config.lr_policy == 'plateau':
                self.dis_scheduler.step(val_loss)

        if self.gen_scheduler is not None:
            if config.lr_policy == 'step':
                self.gen_scheduler.step()
            elif config.lr_policy == 'plateau':
                self.gen_scheduler.step(val_loss)

    def load(self, checkpoint_dir, key, epochs):
        
        # Load generators
        ckpt_name = get_model_ckpt_name(checkpoint_dir, key, epochs)
        assert os.path.isfile(ckpt_name), f"No checkpoint found at {ckpt_name}"
        
        print(f'ContentNet ckpt file name: {ckpt_name}')
        
        state_dict = torch.load(ckpt_name)
        self.gen_a.load_state_dict(state_dict['a'])

    def resume(self, checkpoint_dir, config):
        
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_lr = state_dict['gen_lr']
        epochs = state_dict['epochs']
        
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_lr = state_dict['dis_lr']
        
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, epochs)
        self.gen_scheduler = get_scheduler(self.gen_opt, config, epochs)
        print('Resume from epoch %d' % epochs)
        return epochs

    def save(self, ckpt_dir, epochs):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(ckpt_dir, 'gen_%04d.pt' % (epochs + 1))
        dis_name = os.path.join(ckpt_dir, 'dis_%04d.pt' % (epochs + 1))
        opt_name = os.path.join(ckpt_dir, 'optimizer.pt')
        
        # generators
        torch.save({'a': self.gen_a.state_dict(),
                    'epochs': epochs + 1,
                    'gen_lr': self.gen_opt.param_groups[0]['lr']}, gen_name)
        
        # discriminators
        torch.save({'a': self.dis_a.state_dict(),
                    'epochs': epochs + 1,
                    'dis_lr': self.dis_opt.param_groups[0]['lr']}, dis_name)
        
        # optimizers
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)


