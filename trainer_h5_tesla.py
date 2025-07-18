"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks_tesla import AdaINGen, PatchGAN_Dis
from utils import weights_init, get_model_list, get_scheduler, get_model_ckpt_name
import torch
import torch.nn as nn
import os
from monai.losses.ssim_loss import SSIMLoss
import numpy as np
from monai.transforms import Resize
from networks_tesla import define_F

from trainer_h5_contentnet import ContentNet_Trainer

torch.manual_seed(0)

class TESLA_Trainer(nn.Module):
    def __init__(self, config):
        super(TESLA_Trainer, self).__init__()
        
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
        self.dataset    = config.dataset
        self.workers    = config.workers
        self.epochs     = config.epochs
        self.batch_size = config.batch_size
        self.gen_lr     = config.gen_lr
        self.dis_lr     = config.dis_lr
        self.beta1      = config.beta1
        self.beta2      = config.beta2
        # self.gpu_ids    = config.gpu

        # Test configurations.
        self.test_epochs = config.test_epochs

        # ContentNet(pre-trained)
        if config.train_dataset:
            self.dir_path_contentNet = config.ckpt_dir_ContentNet + config.tb_comment_ContentNet
            self.trainer_ContentNet  = ContentNet_Trainer( config )
            self.trainer_ContentNet.load(checkpoint_dir=self.dir_path_contentNet, key="gen", epochs=100)
        
        #---------- Generator ----------#
        self.gen_b = AdaINGen(self.input_ch_b, config)
        self.netF  = define_F(input_nc     = config.input_ch_b,
                              netF         = config.netF, 
                              norm         = 'batch', 
                              use_dropout  = False, 
                              init_type    = 'normal', 
                              init_gain    = 0.02, 
                              no_antialias = False, 
                              gpu_ids      = config.device, 
                              opt          = config )
        

        #---------- Discriminator ----------#
        self.dis_b = PatchGAN_Dis() # discriminator for domain b
        self.style_dim    = config.gen_style_dim

        # loss
        self.ganloss        = torch.nn.MSELoss()
        self.L1loss         = torch.nn.L1Loss()
        self.SSIMloss       = SSIMLoss(spatial_dims=2, data_range=1)
        self.PatchNCEloss = []

        for nce_layer in config.nce_layers: 
            self.PatchNCEloss.append(PatchNCELoss(config).to(self.device))
            
        # fix the noise used in sampling
        batch_size = int(config.batch_size) 

        # Setup the optimizers
        beta1 = config.beta1
        beta2 = config.beta2

        dis_b_params = list(self.dis_b.parameters())
        gen_b_params = list(self.gen_b.parameters())

        # Optimizers
        self.dis_b_opt = torch.optim.Adam([p for p in dis_b_params if p.requires_grad],
                                        lr=self.dis_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.gen_b_opt = torch.optim.Adam([p for p in gen_b_params if p.requires_grad],
                                        lr=self.gen_lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        
        # Schedulers
        self.dis_b_scheduler     = get_scheduler(self.dis_b_opt, config)
        self.gen_b_scheduler     = get_scheduler(self.gen_b_opt, config)

        # Network weight initialization
        self.apply(weights_init(config.init))
        self.dis_b.apply(weights_init('gaussian'))

        # Miscellaneous
        self.dis_loss_tb  = {}
        self.gen_loss_tb  = {}
        self.prog_4to2_tb = {}
        self.prog_2to1_tb = {}


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def prog_2to1_update(self, config, b_41, b_hr):
        
        if config.pretrained_4to2 == True:
            """
            pretrained progNet_4to1
            """
            self.progNet_4to2.eval()
            self.progNet_2to1_opt.zero_grad()
            recon_b_4to2 = self.progNet_4to2(b_41)
            recon_b_hr   = self.progNet_2to1(recon_b_4to2)

            loss_l1_2to1     = self.L1loss(recon_b_hr, b_hr)
            loss_ssim_2to1   = self.SSIMloss(recon_b_hr, b_hr)
            loss_percep_2to1 = self.Perceptualloss(recon_b_hr, b_hr)

            loss_prog_b_2to1_total = config.recon_l1_w         * loss_l1_2to1 +\
                                config.recon_ssim_w       * loss_ssim_2to1 +\
                                config.recon_perceptual_w * loss_percep_2to1

            loss_prog_b_2to1_total.backward()
            self.progNet_2to1_opt.step()
            
            self.prog_2to1_tb["P_loss/L1_2to1"]         = config.recon_l1_w         * loss_l1_2to1
            self.prog_2to1_tb["P_loss/SSIM_2to1"]       = config.recon_ssim_w       * loss_ssim_2to1
            self.prog_2to1_tb["P_loss/Perceptual_2to1"] = config.recon_perceptual_w * loss_percep_2to1
            
            return loss_prog_b_2to1_total, self.prog_2to1_tb, self.prog_2to1_opt, recon_b_4to2, recon_b_hr
    
    def prog_2to1_forward(self, config, b_41):
        
        if config.pretrained_4to2 and config.pretrained_2to1:
            """
            pretrained progNet_4to1
            """
            self.progNet_4to2.eval()
            self.progNet_2to1.eval()

            recon_b_4to2 = self.progNet_4to2(b_41)
            recon_b_hr   = self.progNet_2to1(recon_b_4to2)

            return recon_b_hr
    
    def dis_update(self, x_b_sr_2to1, config, *data):
        
        canny_edge  = data[-2].float()
        b_4fold = data[-1]

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, x_b_sr_2to1.shape[-2] // 2 ** 4, x_b_sr_2to1.shape[-1] // 2 ** 4) # 

        # Adversarial ground truths
        real = torch.Tensor(np.ones((x_b_sr_2to1.size(0), *patch))).to(self.device)
        fake = torch.Tensor(np.zeros((x_b_sr_2to1.size(0), *patch))).to(self.device)

        self.dis_b_opt.zero_grad()
        # with torch.cuda.amp.autocast():
        # s_a = torch.randn(x_b_sr_2to1.size(0), self.style_dim, 1, 1).to( self.device )
        
        # encode
        c_b_sr_2to1, s_b_sr_2to1_prime = self.gen_b.encode(x_b_sr_2to1)
                                                                        
                                                                        
        # decode
        x_b_sr_2to1_recon = self.gen_b.decode(c_b_sr_2to1[-1], s_b_sr_2to1_prime)

        # D Real loss
        pred_real = self.dis_b( img           = x_b_sr_2to1[:], # (B, 1, 128, 256)
                                img_condition = canny_edge ) 
        loss_real = self.ganloss( pred_real, real )

        # D Fake loss
        pred_fake = self.dis_b( img           = x_b_sr_2to1_recon.detach(), # (B, 1, 128, 256)
                                img_condition = canny_edge )                # (B, 1, 128, 256)
        loss_fake = self.ganloss( pred_fake, fake )
        
        # D Total loss
        self.loss_dis_b_total = 0.5 * ( loss_real + loss_fake )        
        self.loss_dis_total = config.gan_w * (self.loss_dis_b_total)
        
        # D loss for tensorboard visualization
        self.dis_loss_tb['D_loss/dis_adv_b'] = self.loss_dis_total

        self.loss_dis_total.backward()
        self.dis_b_opt.step()
        
        return self.loss_dis_total, self.dis_loss_tb, self.dis_b_opt, pred_real, pred_fake
        
        
    
    def gen_update(self, x_b_sr_2to1, x_b_hr, x_a, config,
                   data_cdt_edge, b_4fold ):

        
        canny_edge  = data_cdt_edge.float()
        
        b_4fold = b_4fold

        """
        Domain 'b': SR T2

        """
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, x_b_sr_2to1.shape[-2] // 2 ** 4, x_b_sr_2to1.shape[-1] // 2 ** 4) 
        
        # Adversarial ground truths
        real = torch.Tensor(np.ones((x_b_sr_2to1.size(0), *patch))).to(self.device)
        fake = torch.Tensor(np.zeros((x_b_sr_2to1.size(0), *patch))).to(self.device)

        self.gen_b_opt.zero_grad()
        # with torch.cuda.amp.autocast():
        s_b_sr_2to1 = torch.randn(x_b_sr_2to1.size(0), self.style_dim, 1, 1).to( self.device )
        
        
        # encode
        c_b_sr_2to1, s_b_sr_2to1_prime = self.gen_b.encode(x_b_sr_2to1)
        
        # decode
        x_b_sr_2to1_recon = self.gen_b.decode(c_b_sr_2to1[-1], s_b_sr_2to1_prime)                   

        # encode again
        c_b_sr_2to1_recon, s_b_sr_2to1_recon = self.gen_b.encode(x_b_sr_2to1_recon)
        
        # decode again (if needed)
        x_b_sr_2to1_recon_b_sr_2to1 = self.gen_b.decode(c_b_sr_2to1_recon[-1], s_b_sr_2to1_recon) if config.recon_x_cyc_w > 0 else None

        ################################################
        ###########  Data consistency term  ############
        ################################################
        src_shape = x_b_sr_2to1_recon.shape
        tgt_shape = b_4fold.shape

        if config.dc_avg:
            data_consistency_1to4 = x_b_sr_2to1_recon.reshape(-1, 1, tgt_shape[-2], int(src_shape[-2]/tgt_shape[-2]), src_shape[-1]).mean(axis=3)
        
        elif config.dc_monai:
            resize_4fold = Resize(spatial_size=(tgt_shape[-2], src_shape[-1]), mode=config.dc_monai_method)
            data_consistency_1to4 = torch.zeros_like(b_4fold)

            for b in range(x_b_sr_2to1_recon.shape[0]):
                data_consistency_1to4[b,:,:,:] = resize_4fold(x_b_sr_2to1_recon[b,:,:,:])

        self.l1loss_data_consistency_1to4   = config.dc_l1_w * self.L1loss(data_consistency_1to4, b_4fold)
        self.ssimloss_data_consistency_1to4 = config.dc_ssim_w * self.SSIMloss(data_consistency_1to4, b_4fold)

        ################################################
        ##################### Loss #####################
        ################################################
        # G Fake loss
        pred_fake = self.dis_b( img           = x_b_sr_2to1_recon,
                                img_condition = canny_edge )
        loss_fake = self.ganloss( pred_fake, real )
        
        # G Total loss
        self.loss_gen_b_total = loss_fake
        self.loss_gen_adv_total = config.gan_w * (self.loss_gen_b_total)

        # pixel-wise reconstruction loss
        # L1 loss
        self.l1loss_gen_recon_x_b_sr_2to1    = config.recon_l1_x_w * self.L1loss(x_b_sr_2to1_recon, x_b_hr[:]) # within-domain recon (sr t2)
        self.l1loss_gen_recon_s_b_sr_2to1    = config.recon_l1_s_w * self.L1loss(s_b_sr_2to1_recon, s_b_sr_2to1_prime) # within-domain style recon (sr t2)
        self.l1loss_gen_recon_c_b_sr_2to1    = config.recon_l1_c_w * self.L1loss(c_b_sr_2to1_recon[-1], c_b_sr_2to1[-1]) # cross-domain content recon (sr t2)
        self.l1loss_gen_cycrecon_x_b_sr_2to1 = config.recon_l1_cyc_w * self.L1loss(x_b_sr_2to1_recon_b_sr_2to1, x_b_sr_2to1[:]) if config.recon_x_cyc_w > 0 else 0
        
        # SSIM loss
        self.ssimloss_gen_recon_x_b_sr_2to1    = config.recon_ssim_x_w * self.SSIMloss(x_b_sr_2to1_recon, x_b_hr[:])
        self.ssimloss_gen_recon_c_b_sr_2to1    = config.recon_ssim_c_w * self.SSIMloss(c_b_sr_2to1_recon[-1], c_b_sr_2to1[-1])
        self.ssimloss_gen_cycrecon_x_b_sr_2to1 = config.recon_ssim_cyc_w * self.SSIMloss(x_b_sr_2to1_recon_b_sr_2to1, x_b_sr_2to1[:]) if config.recon_x_cyc_w > 0 else 0
        
        # PatchNCE loss(contrastive loss)
        if config.nce:
            self.loss_NCE_c_b_sr_2to1 = config.recon_patchnce_w * self.calculate_NCE_loss(config=config, src=x_a, tgt=x_b_sr_2to1, nce_idt=False)
            self.patchnceloss_gen_c_b_sr_2to1 = self.loss_NCE_c_b_sr_2to1
        else:
            self.patchnceloss_gen_c_b_sr_2to1 = 0

        if config.nce_idt and config.recon_patchnce_w > 0:
            self.loss_NCE_c_b_sr_2to1     = config.recon_patchnce_w * self.calculate_NCE_loss(config=config, src=x_a, tgt=x_b_sr_2to1, nce_idt=False)
            self.loss_NCE_idt_c_b_sr_2to1 = config.recon_patchnce_w * self.calculate_NCE_loss(config=config, src=x_b_sr_2to1, tgt=x_b_sr_2to1, nce_idt=True)
            
            self.patchnceloss_gen_c_b_sr_2to1 = 0.5 * (self.loss_NCE_c_b_sr_2to1 + self.loss_NCE_idt_c_b_sr_2to1)
        

        # Reconstruction loss for tensorboard visualization
        # L1 loss
        self.gen_loss_tb['G_loss/L1_recon_x_b_sr_2to1']     = self.l1loss_gen_recon_x_b_sr_2to1   
        self.gen_loss_tb['G_loss/L1_recon_s_b_sr_2to1']     = self.l1loss_gen_recon_s_b_sr_2to1   
        self.gen_loss_tb['G_loss/L1_recon_c_b_sr_2to1']     = self.l1loss_gen_recon_c_b_sr_2to1   
        self.gen_loss_tb['G_loss/L1_recon_cyc_x_b_sr_2to1'] = self.l1loss_gen_cycrecon_x_b_sr_2to1
        
        # SSIM loss
        self.gen_loss_tb['G_loss/SSIM_recon_x_b_sr_2to1']     = self.ssimloss_gen_recon_x_b_sr_2to1   
        self.gen_loss_tb['G_loss/SSIM_recon_c_b_sr_2to1']     = self.ssimloss_gen_recon_c_b_sr_2to1   
        self.gen_loss_tb['G_loss/SSIM_recon_cyc_x_b_sr_2to1'] = self.ssimloss_gen_cycrecon_x_b_sr_2to1
       
        # gan loss for tensorboard viusalization
        self.gen_loss_tb['G_loss/gan_adv_b_sr_2to1'] = self.loss_gen_adv_total
        
        # contrastive loss for tensorboard visualization
        self.gen_loss_tb["G_loss/PatchNCE_recon_c_b_sr_2to1"] = self.patchnceloss_gen_c_b_sr_2to1

        # for data consistency term
        self.gen_loss_tb["G_loss/L1_for_dc_1to4"]   = self.l1loss_data_consistency_1to4
        self.gen_loss_tb["G_loss/SSIM_for_dc_1to4"] = self.ssimloss_data_consistency_1to4

        # total loss
        self.loss_gen_total = self.l1loss_gen_recon_x_b_sr_2to1      + \
                              self.l1loss_gen_recon_s_b_sr_2to1      + \
                              self.l1loss_gen_recon_c_b_sr_2to1      + \
                              self.l1loss_gen_cycrecon_x_b_sr_2to1   + \
                              self.ssimloss_gen_recon_x_b_sr_2to1    + \
                              self.ssimloss_gen_recon_c_b_sr_2to1    + \
                              self.ssimloss_gen_cycrecon_x_b_sr_2to1 + \
                              self.loss_gen_adv_total        + \
                              self.patchnceloss_gen_c_b_sr_2to1 + \
                              self.l1loss_data_consistency_1to4 + \
                              self.ssimloss_data_consistency_1to4

        self.loss_gen_total.backward()
        self.gen_b_opt.step()

        return self.loss_gen_total, self.gen_loss_tb, self.gen_b_opt, c_b_sr_2to1, x_b_sr_2to1_recon

    def update_learning_rate(self, config, val_loss=None):
        if self.dis_b_scheduler is not None:
            if config.lr_policy == 'step':
                self.dis_b_scheduler.step()
            elif config.lr_policy == 'plateau':
                self.dis_b_scheduler.step(val_loss)

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
        self.gen_b.load_state_dict(state_dict['b'])

    def resume(self, checkpoint_dir, config):
        
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_b.load_state_dict(state_dict['b'])
        self.gen_lr = state_dict['gen_lr']
        epochs = state_dict['epochs']
        
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_b.load_state_dict(state_dict['b'])
        self.dis_lr = state_dict['dis_lr']
        
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_b_opt.load_state_dict(state_dict['dis'])
        self.gen_b_opt.load_state_dict(state_dict['gen'])
        
        # Reinitilize schedulers
        self.dis_b_scheduler = get_scheduler(self.dis_b_opt, config, epochs)
        self.gen_b_scheduler = get_scheduler(self.gen_b_opt, config, epochs)
        print('Resume from epoch %d' % epochs)
        return epochs

    def save(self, ckpt_dir, epochs):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(ckpt_dir, 'gen_%04d.pt' % (epochs + 1))
        dis_name = os.path.join(ckpt_dir, 'dis_%04d.pt' % (epochs + 1))
        opt_name = os.path.join(ckpt_dir, 'optimizer.pt')
        
        # generators
        torch.save({'b': self.gen_b.state_dict(),
                    'epochs': epochs + 1,
                    'gen_lr': self.gen_b_opt.param_groups[0]['lr']}, gen_name)
        
        # discriminators
        torch.save({'b': self.dis_b.state_dict(),
                    'epochs': epochs + 1,
                    'dis_lr': self.dis_b_opt.param_groups[0]['lr']}, dis_name)
        
        # optimizers
        torch.save({'gen': self.gen_b_opt.state_dict(),
                    'dis': self.dis_b_opt.state_dict()}, opt_name)


    def calculate_NCE_loss(self, config, src, tgt, nce_idt:bool = False):
        
        if nce_idt:

            feat_q_c_b, _ = self.gen_b.encode(tgt)

            n_layers = len(feat_q_c_b)
            feat_k_c_a, _ = self.gen_b.encode(src) # pre-trained ContentNet
            feat_k_c_a_pool, sample_ids = self.netF(feat_k_c_a, config.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q_c_b, config.num_patches, sample_ids)

            total_nce_loss = 0.0
            for f_q, f_k, crit in zip(feat_q_pool, feat_k_c_a_pool, self.PatchNCEloss):
                loss = crit(f_q, f_k) * config.lambda_NCE
                total_nce_loss += loss.mean()

            return total_nce_loss / n_layers

        # TESLA setting
        else:

            feat_q_c_b, _ = self.gen_b.encode(tgt)

            n_layers = len(feat_q_c_b) 
            self.trainer_ContentNet.eval()
            feat_k_c_a, _ = self.trainer_ContentNet.gen_a.encode(src) # pre-trained ContentNet
            feat_k_c_a_pool, sample_ids = self.netF(feat_k_c_a, config.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q_c_b, config.num_patches, sample_ids)

            total_nce_loss = 0.0
            for f_q, f_k, crit in zip(feat_q_pool, feat_k_c_a_pool, self.PatchNCEloss):
                loss = crit(f_q, f_k) * config.lambda_NCE
                total_nce_loss += loss.mean()

            return total_nce_loss / n_layers
    
from packaging import version
class PatchNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.config.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.config.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.config.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss