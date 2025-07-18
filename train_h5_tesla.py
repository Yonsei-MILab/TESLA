#%%
import os
import sys
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from trainer_h5_tesla import ProTPSR_Trainer
from customdataset_h5_tesla import IXI_Dataset

import torch
import numpy as np
from torch.backends import cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from eval_h5_tesla import eval_net

logging.basicConfig( level=logging.INFO, format='%(levelname)s: %(message)s' )


#-----------------------------------------------------------------------------------------------------------------------

def cleanup():
    dist.destroy_process_group()

def build_tensorboard(config):
    """Build a tensorboard logger."""
    from logger import Logger
    writer = Logger( config )
    return writer

def save(ckpt_dir, model, optimizer, epochs, stage):

    # Save generators, discriminators, and optimizers
    ProgNet_name = os.path.join(ckpt_dir, f'ProgNet_{stage}_{(epochs + 1):04d}.pt')
    opt_name = os.path.join(ckpt_dir, 'optimizer.pt')
    
    # Stacked_dusunet
    torch.save({'model': model.state_dict(),
                'lr': optimizer.param_groups[0]['lr']}, ProgNet_name)
    
    # Optimizers
    torch.save({'optimizer': optimizer.state_dict()}, opt_name)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if models is None:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name
#-----------------------------------------------------------------------------------------------------------------------

def get_model_ckpt_name(checkpoint_dir, key:str, epochs):
    if os.path.exists(checkpoint_dir) is False:
        return None
    assert os.path.isfile(os.path.join(checkpoint_dir, key + f'_{epochs:04d}.pt')), '해당 ckpt 파일은 존재하지 않습니다.'

    ckpt_name = os.path.join(checkpoint_dir, key + f'_{epochs:04d}.pt')
    return ckpt_name

#-----------------------------------------------------------------------------------------------------------------------

def main(config):

    # For fast training.
    cudnn.benchmark = True

    # # Define usable gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.dataset in ['IXI']:
        train_dataset = IXI_Dataset( config=config, mode='train' )
        test_dataset  = IXI_Dataset( config=config, mode='test' )
        nb_total_train_imgs = len(train_dataset)
        nb_total_test_imgs  = len(test_dataset)
    
    logging.info(f'''Starting training:
        Epochs            : {config.epochs}
        Batch size        : {config.batch_size}
        Learning rate of P: {config.prog_lr}
        Learning rate of G: {config.gen_lr}
        Learning rate of D: {config.dis_lr}
        Training size     : {nb_total_train_imgs}
        Test size         : {nb_total_test_imgs}
        Device            : {config.gpu}
    ''')
    main_worker( config )


def main_worker( config ):
    
    torch.manual_seed(0)

    # Data loader.
    ixi_loader  = None

    if config.dataset in ['IXI']:
        train_dataset = IXI_Dataset( config=config, mode='train' )
        test_dataset  = IXI_Dataset( config=config, mode='test' )

        nb_total_train_imgs = len(train_dataset)
    

    # Set up the tensorboard 
    if config.use_tensorboard:
        writer = build_tensorboard(config)

    
    # Load experiment setting
    max_epoch = config.epochs
    # display_size = config.display_size # How many images do you want to display each time

    # Setup devices for this process
    device = torch.device("cuda:" + config.device)

    # Model
    trainer = ProTPSR_Trainer( config ).to( device )

    # Dataloader
    train_loader = DataLoader( dataset     = train_dataset, 
                               batch_size  = config.batch_size, 
                               shuffle     = True,
                               num_workers = config.workers,
                               pin_memory  = True, 
                            #    sampler     = train_sampler, 
                               drop_last   = True )
    
    eval_loader = DataLoader( dataset     = test_dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = True,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )

    test_loader = DataLoader( dataset     = test_dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = False,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )
    
    # train_loader_ContentNet = DataLoader( dataset     = train_dataset_ContentNet, 
    #                                       batch_size  = config.batch_size, 
    #                                       shuffle     = True,
    #                                       num_workers = config.workers,
    #                                       pin_memory  = True, 
    #                                       #    sampler     = train_sampler, 
    #                                       drop_last   = True )
    
    
    
    # Automatically resume from checkpoint if it exists    
    # ProgNet    
    # dir_path_ProgNet = config.ckpt_dir_ProgNet + config.tb_comment
    # if os.path.isdir(dir_path_ProgNet):
    #     start_epochs = trainer.resume(dir_path_ProgNet, config)
    # else:
    #     start_epochs = 0
    
    start_epochs = 0
    print('start_epoch:', start_epochs + 1)
    
    # Start traing process ---------------------------------------------------------------------------
    for epochs in range( start_epochs, max_epoch ):
        # torch.autograd.set_detect_anomaly(True)
        """
        In distributed mode, calling the set_epoch() method at the beginning of each epoch
        before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        Otherwise, the same ordering will be always used.        
        """
       
        # ContentNet loss 
        dis_total_running_loss  = 0
        gen_total_running_loss  = 0
        dis_total_epoch_loss    = 0
        gen_total_epoch_loss    = 0
        
    
        dict_dis_indiv_running_loss_tb  = {}
        dict_gen_indiv_running_loss_tb  = {}
        dict_dis_indiv_epoch_loss_tb    = {}
        dict_gen_indiv_epoch_loss_tb    = {}
        
        dict_total_epoch_loss  = {}
        
        # For visualizing all the losses utilized in training phase
        dict_merged_epoch_loss_tb = {}
        
        # step size of the generator and discriminator
        generator_steps = config.generator_steps
        discriminator_steps = config.discriminator_steps
        
        # Manually control on tqdm() updates by using a with statement
        with tqdm(total=nb_total_train_imgs,
                  desc=f'Epoch {epochs + 1}/{max_epoch}',
                  unit='imgs',
                  ncols=150,
                  ascii=' ==' ) as pbar:
            
            for step, img in enumerate( train_loader, start=epochs * nb_total_train_imgs):
                
                img["data_A"]         = img["data_A"].to(device)
                img["data_B_41"]      = img["data_B_41"].to( device )
                img["data_B_21"]      = img["data_B_21"].to( device )
                img["data_B_HR"]      = img["data_B_HR"].to( device ) # ground truth t2
                img["data_B_SR_2to1"] = img["data_B_SR_2to1"].to( device ) # super-resolved t2
                img["data_cdt_edge"]  = img["data_cdt_edge"].to( device )
                img["data_B_4fold"]   = img["data_B_4fold"].to( device )
                img["data_PD"]        = img["data_PD"].to( device )

                pd_hr     = img["data_PD"]
                a_hr      = img["data_A"]
                b_41      = img["data_B_41"]    # (B, 1, 128, 256)
                b_21      = img["data_B_21"]    # (B, 1, 128, 256)                
                b_hr      = img["data_B_HR"]    # (B, 1, 128, 256)
                b_4fold   = img["data_B_4fold"]
                b_2fold   = img["data_B_2fold"]
                
                b_sr_2to1 = img["data_B_SR_2to1"]
                data_cdt_edge = img["data_cdt_edge"] # (B, 1, 128, 256)
                

                # =================================================================================== #
                #                           1. Train the Discriminator                                #
                # =================================================================================== #
                
                for _ in range(discriminator_steps):

                    # Discriminator
                    loss_dis_total, dis_indiv_loss_tb, dis_b_opt, pred_real, pred_fake = trainer.dis_update( b_sr_2to1,
                                                                                                             config,
                                                                                                             data_cdt_edge,
                                                                                                             b_4fold )
                    dis_total_running_loss += loss_dis_total

                    for key, val in dis_indiv_loss_tb.items(): 
                        if key not in dict_dis_indiv_running_loss_tb:
                            dict_dis_indiv_running_loss_tb[key] = val
                        else:
                            dict_dis_indiv_running_loss_tb[key] += val

                # # =================================================================================== #
                # #                           2. Train the Generator                                    #
                # # =================================================================================== #

                for _ in range(generator_steps):

                    # Generator
                    if config.hr_pd:

                        loss_gen_total , gen_indiv_loss_tb, gen_b_opt, c_b_sr_2to1, x_b_sr_2to1_recon = trainer.gen_update( b_sr_2to1,
                                                                                                                        b_hr,
                                                                                                                        pd_hr,
                                                                                                                        config,
                                                                                                                        data_cdt_edge,
                                                                                                                        b_4fold )
                    else:
                        loss_gen_total , gen_indiv_loss_tb, gen_b_opt, c_b_sr_2to1, x_b_sr_2to1_recon = trainer.gen_update( b_sr_2to1,
                                                                                                                        b_hr,
                                                                                                                        a_hr,
                                                                                                                        config,
                                                                                                                        # coeffs_hf_comp_B_SR_2to1_1st,
                                                                                                                        # coeffs_hf_comp_B_SR_2to1_2nd,
                                                                                                                        data_cdt_edge,
                                                                                                                        b_4fold )

                    gen_total_running_loss += loss_gen_total
                    
                    for key, val in gen_indiv_loss_tb.items():
                        if key not in dict_gen_indiv_running_loss_tb:
                            dict_gen_indiv_running_loss_tb[key] = val
                        else:
                            dict_gen_indiv_running_loss_tb[key] += val
                
                
                
                
                torch.cuda.synchronize() # 딥러닝을 돌리다보면 GPU는 아직 계산중인데
                                            # CPU는 GPU를 기다리지 않고 그 다음 코드를 실행하려고 할때가 있음
                                            # 즉, GPU와 CPU의 계산 타이밍이 어긋나는 것
                                            # torch.cuda.synchronize()는 그럴때 GPU와 CPU의 타이밍을 맞추기 위해
                                            # GPU 계산이 끝날때까지 CPU execution을 block한다.
            
                pbar.update(img['data_B_HR'].shape[0]) # tqdm의 progress bar를 input data의 (batch_size) * (사용하는 gpu개수) 만큼 
                                                       # 업데이트 해준다.

            
            # Update learning rate
            trainer.dis_b_scheduler.step()
            trainer.gen_b_scheduler.step()
            
            # Comput epoch loss for ProGNet
            # prog_total_epoch_loss = prog_b_2to1_total_running_loss / (nb_total_train_imgs)
            dis_total_epoch_loss = dis_total_running_loss / (nb_total_train_imgs*discriminator_steps)
            gen_total_epoch_loss = gen_total_running_loss / (nb_total_train_imgs*generator_steps)
            
            dict_total_epoch_loss = { 'D_loss/total': dis_total_epoch_loss,
                                      'G_loss/total': gen_total_epoch_loss }

            # for key, val in dict_prog_indiv_running_loss_tb.items():
            #     dict_prog_indiv_epoch_loss_tb[key] = val / nb_total_train_imgs

            for key, val in dict_dis_indiv_running_loss_tb.items():
                dict_dis_indiv_epoch_loss_tb[key] = val / nb_total_train_imgs
            
            for key, val in dict_gen_indiv_running_loss_tb.items():
                dict_gen_indiv_epoch_loss_tb[key] = val / nb_total_train_imgs
                

            # Dict type의 변수를 합치는 과정
            dict_merged_epoch_loss_tb.update( dict_total_epoch_loss )
            dict_merged_epoch_loss_tb.update( dict_dis_indiv_epoch_loss_tb )
            dict_merged_epoch_loss_tb.update( dict_gen_indiv_epoch_loss_tb )

            # Save checkpoint per 5 epochs
            if (epochs + 1) % config.model_save_step == 0:
                
                # Create directories if not exist.
                
                # ProgDRL
                if not os.path.exists(os.path.join(config.ckpt_dir_tesla, config.tb_comment)):
                    os.makedirs(os.path.join(config.ckpt_dir_tesla, config.tb_comment), exist_ok=True)
                                    
                
                # ProgDRL
                trainer.save( os.path.join(config.ckpt_dir_tesla, config.tb_comment), epochs )


            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            if (epochs + 1) % config.img_display_step == 0: 
                
                # Visualization of result images during training
                with torch.no_grad():
                    assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
                       
                    # Visualize loss graphs
                    for tag, value in dict_merged_epoch_loss_tb.items():
                        # print(f"Type of value: {type(value)}")
                        print(f"\n{tag}: {value:.4f}")
                        # if isinstance(value, monai.data.meta_tensor.MetaTensor):
                            # value = torch.Tensor(value.cpu())  # Convert MetaTensor to torch.Tensor
                        writer.scalar_summary(tag, value.item(), epochs+1)  # Use .item() to get a Python number from a tensor containing a single value
                        
                    # writer.scalar_summary('P_loss/lr', prog_2to1_opt.param_groups[0]['lr'], epochs + 1)
                    writer.scalar_summary('G_loss/lr', gen_b_opt.param_groups[0]['lr'], epochs + 1)
                    writer.scalar_summary('D_loss/lr', dis_b_opt.param_groups[0]['lr'], epochs + 1)
 
            # Validation phase
            if (epochs + 1) % 1 == 0:                
                eval_net( trainer, eval_loader, config, epochs, device, writer )

            if (epochs + 1) == 2:   
                log_command(config)

        # trainer.update_learning_rate(config)
    writer.writer.close()
    
    # cleanup()
#-----------------------------------------------------------------------------------------------------------

def str2bool(v):
    return v.lower() in ('true')

def log_command(args):

    # log file directory
    log_file_path = 'train_tesla.txt'

    if not os.path.exists(log_file_path):
        open(log_file_path, 'w').close()  # Make an empty file

    # Current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Command line
    command = '/home/milab/anaconda3/envs/yoonseok/bin/python /SSD4_8TB/CYS/02.Super-resolution_Missing_data_imputation/04.ProTPSR/TESLA/train_h5_tesla.py ' + ' '.join(sys.argv[1:])

    # Leave the log
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{current_time} : {command}\n\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Through Plane Super Res')
    # current_date = datetime.date.today()

    # Training Configurations
    parser.add_argument('-gpu', '--gpu',             type=str,      default="7")
    parser.add_argument('-device', '--device',       type=str,      default="0")
    parser.add_argument('--dataset',                 type=str,      default='IXI', choices=['IXI'])
    parser.add_argument('--data_range',              type=float,    default=1,      help='Data range setting of SSIM')
    parser.add_argument('-w', '--workers',           type=int,      default=8,         help='number of data loader workers')
    parser.add_argument('-e', '--epochs',            type=int,      default=1,         help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size',        type=int,      default=10,        help='mini-batch size')
    parser.add_argument('-g_lr', '--gen_lr',         type=float,    default=1e-4,      help='Learning rate of G')
    parser.add_argument('-d_lr', '--dis_lr',         type=float,    default=1e-4,      help='Learning rate of D')
    parser.add_argument('-p_lr', '--prog_lr',        type=float,    default=1e-4,      help='Learning rate of P')
    parser.add_argument('--stacked_dsunet_lr',       type=float,    default=1e-4, help='Learning rate of Stacked_DSUNet')
    parser.add_argument('--beta1',                   type=float,    default=0.5,       help='beta1 for Adam optimizer')
    parser.add_argument('--beta2',                   type=float,    default=0.999,     help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay',            type=float,    default=1e-4,      help='weight decay for Adam optimizer')
    parser.add_argument('--init',                    type=str,      default='kaiming', help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr_policy',               type=str,      default='step',    help='learning rate scheduler')
    parser.add_argument('--step_size',               type=int,      default=20,        help='how often to decay learning rate when using step scheduler')
    parser.add_argument('--gamma',                   type=float,    default=0.1,       help='how much to decay learning rate when using StepLR scheduler')
    parser.add_argument('--patience',                type=int,      default=10,        help='how much to be patient for decaying learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--factor',                  type=float,    default=0.1,       help='how much to decay learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--gan_w',                   type=float,    default=1.0,         help='weight of adversarial loss')
    parser.add_argument('--recon_x_w',               type=float,    default=1.0,         help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w',               type=float,    default=1.0,         help='weight of style reconstruction loss')
    parser.add_argument('--recon_style_match_w',     type=float,    default=1.0,         help='weight of style reconstruction loss using gram matrix')
    parser.add_argument('--recon_c_w',               type=float,    default=1.0,         help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w',           type=float,    default=1.0,         help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--vgg_w',                   type=float,    default=1.0,         help='weight of domain-invariant perceptual loss')
    parser.add_argument('--dc_l1_w',                 type=float,    default=1.0,         help='weight of l1 loss for data consistency term')
    parser.add_argument('--dc_ssim_w',               type=float,    default=1.0,         help='weight of ssim loss for data consistency term')
    parser.add_argument('--recon_l1_x_w',            type=float,    default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_l1_s_w',            type=float,    default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_l1_c_w',            type=float,    default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_l1_cyc_w',          type=float,    default=1.0,         help='weight of l1 loss')
    parser.add_argument('--recon_ssim_x_w',          type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_c_w',          type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_cyc_w',        type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_x_a',        type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_c_a',        type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_cyc_x_a',    type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_c_a_dwt_1st',type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_ssim_w_c_a_dwt_2nd',type=float,    default=1.0,         help='weight of ssim loss')
    parser.add_argument('--recon_patchnce_w',        type=float,    default=1.0,         help='weight of  patchnce loss')
    parser.add_argument('--generator_steps',         type=int,      default=1,         help='number of steps for generator training')
    parser.add_argument('--discriminator_steps',     type=int,      default=1,         help='number of steps for discriminator training')
    parser.add_argument('--domain_b',                type=str2bool, default=False,     help='Domain of input image')
    parser.add_argument('--dc_avg',                  type=str2bool, default=False,     help='The way of Data consistency term')
    parser.add_argument('--dc_monai',                type=str2bool, default=False,     help='The way of Data consistency term')
    parser.add_argument('--dc_monai_method',         type=str,      default="area", choices=["area", "linear", "nearest"],  help='The way of interpolation for Data consistency term in monai')


    # Test configurations
    parser.add_argument('--test_epochs', type=int, default=10, help='test model from this epoch')

    # Data options
    parser.add_argument('--input_ch_a', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--input_ch_b', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--hr_pd',   type=str2bool,  default=False,   help='whether input PD is used or not')
    parser.add_argument('--crf_domain', type=str, choices=["t1", "pd", "t2","srt2","none"], required=True)
    
    parser.add_argument('-nb_train_imgs', '--nb_train_imgs', type=int, default=20, help='number of images employed for train code checking')
    parser.add_argument('-nb_test_imgs', '--nb_test_imgs', type=int, default=20, help='number of images employed for test code checking')
    parser.add_argument('--train_dataset', type=str2bool, default=None,  help='Condition whether test the trained network on train dataset')

    # Model configurations
    # Generator
    parser.add_argument('--gen_dim',          type=int, default=64,        help='number of filters in the bottommost layer = the # of channels of content code')
    parser.add_argument('--gen_mlp_dim',      type=int, default=256,       help='number of filters in MLP')
    parser.add_argument('--gen_style_dim',    type=int, default=8,         help='length of style code')
    parser.add_argument('--gen_activ',        type=str, default='relu',    help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--gen_n_downsample', type=int, default=2,         help='number of downsampling layers in content encoder')
    parser.add_argument('--gen_n_res',        type=int, default=4,         help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--gen_pad_type',     type=str, default='reflect', help='padding type [zero/reflect]')

    # Discriminator
    parser.add_argument('--dis_dim',        type=int, default=64,        help='number of filters in the bottommost layer')
    parser.add_argument('--dis_norm',       type=str, default='none',    help='normalization layer [none/bn/in/ln]')
    parser.add_argument('--dis_activ',      type=str, default='lrelu',   help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--dis_n_layer',    type=int, default=4,         help='number of layers in D')
    parser.add_argument('--dis_gan_type',   type=str, default='lsgan',   help='GAN loss [lsgan/nsgan]')
    parser.add_argument('--dis_num_scales', type=int, default=3,         help='number of scales')
    parser.add_argument('--dis_pad_type',   type=str, default='reflect', help='padding type [zero/reflect]')
    parser.add_argument('--sr_scale', type=int, required=True)
    
    # About PatchNCE loss
    parser.add_argument('--lambda_GAN',  type=float,    default=1.0, help='weight for GAN loss：GAN(G(X))')
    parser.add_argument('--lambda_NCE',  type=float,    default=1.0, help='weight for NCE loss: NCE(G(X), X)')
    parser.add_argument('--nce',     type=str2bool,     default=True, nargs='?', const=True,  help='use NCE loss')
    parser.add_argument('--nce_idt',     type=str2bool, default=False, nargs='?', const=True,  help='use NCE loss for identity mapping: NCE(G(Y), Y))')
    parser.add_argument('--nce_layers',  type=str,      default='0,1,2', help='compute NCE loss on which layers')
    parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                        type=str2bool, nargs='?',       default=False, const=True,
                        help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
    parser.add_argument('--netF',        type=str,      default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
    parser.add_argument('--netF_nc',     type=int,      default=256)
    parser.add_argument('--nce_T',       type=float,    default=0.07, help='temperature for NCE loss')
    parser.add_argument('--num_patches', type=int,      default=256,  help='number of patches per layer')
    parser.add_argument('--flip_equivariance',
                        type=str2bool, nargs='?',       default=False, const=True, 
                        help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

    # Directories
    
    # ProgDRL
    parser.add_argument('--log_dir',              type=str,  default='ProTPSR_DRL/logs/ProgDRL/')
    parser.add_argument('--ckpt_dir_tesla',       type=str,  default='ProTPSR_DRL/ckpts/ProgDRL/', help='path to checkpoint directory')
    parser.add_argument('--pretrained_4to2',      type=bool, default=True, help='Options for whether to use pre-trained models')
    parser.add_argument('--pretrained_2to1',      type=bool, default=True, help='Options for whether to use pre-trained models')
    
    # ContentNet
    parser.add_argument('--ckpt_dir_ContentNet',  type=str,  default='tesla_contentnet/ckpts/ContentNet/', help='path to checkpoint directory')

    # Data dir
    parser.add_argument('--ixi_h5_1_2mm_dir', type=str, default='ProTPSR/outputs/ProgNet_2to1/_ProgNet_2024_08_28_ixi_ProgNet_only_2to1_resized_input_norm_0to1_l1_ssim_1_lr_0.0001_train_n_test_4000_10_b_10_e_50/')
    
    
    # Miscellaneous
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--tb_display_size', type=int, default=3)
    parser.add_argument('--tb_comment', type=str, default='_TPSR_2023_10_25_wo_perceptual_loss_on_xab_t1_1.2mm_t2_2mm_b_16_e_1000/')
    parser.add_argument('--tb_comment_ContentNet', type=str, default="_ProgNet_2024_09_02_ixi_ContentNet_wo_dwt_total_1ch_input_norm_0to1_l1_ssim_train_n_test_4000_10_b_5_e_100")
    parser.add_argument('--tb_comment_4to2', type=str, default="_ProgNet_2024_08_28_ixi_ProgNet_only_4to2_resized_input_norm_0to1_l1_ssim_1_lr_0.0001_train_n_test_4000_10_b_10_e_50/")
    parser.add_argument('--tb_comment_2to1', type=str, default="_ProgNet_2024_08_28_ixi_ProgNet_only_2to1_resized_input_norm_0to1_l1_ssim_1_lr_0.0001_train_n_test_4000_10_b_10_e_50/")

    # Step size.
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--img_save_step', type=int, default=2)
    parser.add_argument('--img_display_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=5)
    # parser.add_argument('--lr_update_step', type=int, default=1000)

   

    parser.set_defaults()  # no image pooling

    config = parser.parse_args()
    # print(config)
    main(config)