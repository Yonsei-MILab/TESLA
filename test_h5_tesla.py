import os
import sys
import h5py
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from trainer_h5_tesla import TESLA_Trainer
from customdataset_h5_tesla import IXI_Dataset
from monai.transforms import Resize

import torch
import numpy as np
from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric

logging.basicConfig( level=logging.INFO, format='%(levelname)s: %(message)s' )

#-----------------------------------------------------------------------------------------------------------------------

def cleanup():
    dist.destroy_process_group()

def build_tensorboard_test(config):
    """Build a tensorboard logger."""
    from logger_test import Logger
    writer = Logger( config )
    return writer

def log_command(args):
    
    # log file directory
    log_file_path = 'test_tesla.txt'
    
    if not os.path.exists(log_file_path):
        open(log_file_path, 'w').close()  # Make an empty file
    
    # Current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Command line
    command = '/home/milab/anaconda3/envs/yoonseok/bin/python /SSD4_8TB/CYS/02.Super-resolution_Missing_data_imputation/04.ProTPSR/TESLA/test_h5_tesla.py ' + ' '.join(sys.argv[1:])

    # Leave the log
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{current_time} : {command}\n\n')

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
    assert os.path.isfile(os.path.join(checkpoint_dir, key + '_0' + str(epochs) + '.pt')), '해당 ckpt 파일은 존재하지 않습니다.'

    ckpt_name = os.path.join(checkpoint_dir, key + '_0' + str(epochs) + '.pt')
    return ckpt_name

def denorm(img):
    denorm_img = img*0.5 + 0.5
    return denorm_img

def minmaxnorm(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img
    
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------

def main(config):

    # For fast training.
    cudnn.benchmark = True

    # # Define usable gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.dataset in ['IXI']:
        test_dataset  = IXI_Dataset( config=config, mode='test' )
        nb_total_test_imgs  = len(test_dataset)
    
    logging.info(f'''Starting training:
        Epochs            : {config.epochs}
        Batch size        : {config.batch_size}
        Learning rate of G: {config.gen_lr}
        Learning rate of D: {config.dis_lr}
        Test size         : {nb_total_test_imgs}
        Device            : {config.gpu}
    ''')
    main_worker( config )


def main_worker( config ):
    """
    main worker안에
    Distributed Data Parallel (DDP)
    set up부터 다 되어있다.
    
    """

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Make dir for test output
    base_dir = 'outputs/tesla/' + config.tb_comment + "/test/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # Define a TestDataset
    if config.dataset in ['IXI']:
        # train_dataset  = IXI_Dataset( config=config, mode='train' )
        test_dataset   = IXI_Dataset( config=config, mode='test' )
    

    # Set up the tensorboard 
    if config.use_tensorboard:
        writer = build_tensorboard_test(config)

    # Setup devices for this process
    device = torch.device("cuda:" + config.device)

    # Model
    trainer = TESLA_Trainer( config ).to( device )
        
    # Dataloader    
    test_loader = DataLoader( dataset     = test_dataset, 
                              batch_size  = config.batch_size, 
                              shuffle     = False,
                              num_workers = 4,
                              pin_memory  = True, 
                            #   sampler     = test_sampler, 
                              drop_last   = True )
    
    # epochs = 200
    
    #------------------- Load the weights of model from checkpoint if it exists -------------------------------------#
    # ContentNet 
    dir_path_tesla = config.ckpt_dir_tesla + config.tb_comment

    print(f'tesla_ckpt directory: {dir_path_tesla}\n\n')

    trainer.load(checkpoint_dir=dir_path_tesla, key="gen", epochs=config.epochs)
    n_test = len(test_loader)   
 
    # data_A
    data_A_list = []

    # data_PD
    data_PD_list = []

    # data_B
    data_B_HR_list    = []
    data_B_4to1_list  = []
    data_B_2to1_list  = []
    data_B_4fold_list = []
    data_B_2fold_list = []

    # data_B_recon
    data_B_SR_2to1_list = []
    data_B_recon_list   = []

    # c_a
    c_b_sr_2to1_ds_1st_list = []
    c_b_sr_2to1_ds_2nd_list = []
    c_b_sr_2to1_final_list  = []

    # canny edge
    canny_edge_list = []

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1)
    psnr_metric = PSNRMetric(max_val=1)

    tesla_running_ssim_value = 0
    tesla_running_psnr_value = 0
    tesla_mean_ssim_value = 0
    tesla_mean_psnr_value = 0
    n_samples = 0

    dict_tesla_mean_metric_tb = {}
    
    with tqdm(total=n_test, 
              desc='Test', 
              unit='imgs', 
              ncols=180,
              ascii=' ==') as pbar:
        with torch.no_grad():
            # assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
            # """
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch
            # before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.        
            # """
            # # test_sampler.set_epoch( epochs )
                        
            for step, img in enumerate(test_loader, len(test_loader)):
                
                img["data_A"]         = img["data_A"].to( device )
                img["data_B_SR_2to1"] = img["data_B_SR_2to1"].to( device ) # super-resolved t2 from PR stage
                img["data_B_HR"]      = img["data_B_HR"].to( device )
                img["data_PD"]        = img["data_PD"].to(device)

                a_hr      = img["data_A"]
                b_sr_2to1 = img["data_B_SR_2to1"]
                b_hr      = img["data_B_HR"]
                b_2fold   = img["data_B_2fold"] # (B, 1, 64, 256)
                b_4fold   = img["data_B_4fold"] # (B, 1, 32, 256)
                b_21      = img["data_B_21"]
                b_41      = img["data_B_41"]

                trainer.eval()
                c_b_sr_2to1, s_b_sr_2to1_prime = trainer.gen_b.encode( b_sr_2to1 )
                x_b_sr_2to1_recon = trainer.gen_b.decode( c_b_sr_2to1[-1], s_b_sr_2to1_prime )
                trainer.train()

                # Compute SSIM and PSNR
                ssim_value = ssim_metric(y_pred=x_b_sr_2to1_recon, y=b_hr)
                psnr_value = psnr_metric(y_pred=x_b_sr_2to1_recon, y=b_hr)
                
                # pdb.set_trace()
                tesla_running_ssim_value += torch.sum(ssim_value).item()
                tesla_running_psnr_value += torch.sum(psnr_value).item()
                
                n_samples += b_hr.size(0)
                
                # data_A
                data_A_list.append(img["data_A"][:].cpu().numpy())
                data_PD_list.append(img["data_PD"][:].cpu().numpy())

                # data_B
                data_B_HR_list.append(img["data_B_HR"][:].cpu().numpy())
                data_B_4to1_list.append(b_41.numpy())
                data_B_2to1_list.append(b_21.numpy())
                data_B_4fold_list.append(b_4fold.numpy())
                data_B_2fold_list.append(b_2fold.numpy())
                data_B_SR_2to1_list.append(b_sr_2to1.cpu().numpy())

                # data_B
                data_B_recon_list.append(x_b_sr_2to1_recon.cpu().numpy())

                # c_b
                c_b_sr_2to1_ds_1st_list.append(c_b_sr_2to1[0].cpu().numpy())
                c_b_sr_2to1_ds_2nd_list.append(c_b_sr_2to1[1].cpu().numpy())
                c_b_sr_2to1_final_list.append(c_b_sr_2to1[-1].cpu().numpy())

                pbar.update(img['data_B_HR'].shape[0])

                # =================================================================================== #
                #                                 3. Miscellaneous                                    #
                # =================================================================================== #

                # For TESLA
                tb_a_hr               = torch.Tensor(a_hr[:config.tb_display_size].cpu())
                tb_c_b_sr_2to1_ds_1st = torch.Tensor(c_b_sr_2to1[0][:config.tb_display_size, 0:1].cpu())  
                tb_c_b_sr_2to1_ds_2nd = torch.Tensor(c_b_sr_2to1[1][:config.tb_display_size, 0:1].cpu())  
                tb_c_b_sr_2to1_final  = torch.Tensor(c_b_sr_2to1[-1][:config.tb_display_size, 0:1].cpu()) 
                tb_b_sr_2to1          = torch.Tensor(b_sr_2to1[:config.tb_display_size].cpu())
                tb_recon_b_sr_2to1    = torch.Tensor(x_b_sr_2to1_recon[:config.tb_display_size].cpu())  # config.tb_display_size, 1, 128, 256
                tb_b_hr               = torch.Tensor(b_hr[:config.tb_display_size].cpu())               # config.tb_display_size, 1, 128, 256
                
                resize_transform = Resize(spatial_size=b_hr.shape[2:]) # Resize(spatial_size=(128,256))

                resize_c_b_sr_2to1_ds_1st = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_ds_1st[i][:]) for i in range(tb_c_b_sr_2to1_ds_1st.shape[0])]))
                resize_c_b_sr_2to1_ds_2nd = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_ds_2nd[i][:]) for i in range(tb_c_b_sr_2to1_ds_2nd.shape[0])]))
                resize_c_b_sr_2to1_final  = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_final[i][:]) for i in range(tb_c_b_sr_2to1_final.shape[0])]))  
                
                tesla_img_tensor = torch.cat([resize_c_b_sr_2to1_ds_1st,
                                                resize_c_b_sr_2to1_ds_2nd,
                                                resize_c_b_sr_2to1_final,
                                                tb_b_sr_2to1,
                                                tb_recon_b_sr_2to1,
                                                tb_b_hr,
                                                tb_a_hr], dim=0)
                
                tesla_img_grid = torchvision.utils.make_grid( tensor    = tesla_img_tensor.data,
                                                            nrow      = config.tb_display_size,
                                                            padding   = 0,
                                                            normalize = True )

                writer.image_summary( tag        = 'test/tesla--cbDs1st--cbDs2nd--cbFinal--xb_sr--xbsrRecon--xb_hr--xa_hr',
                                    img_tensor = tesla_img_grid,
                                    step       = step + 1 )

            tesla_mean_ssim_value = tesla_running_ssim_value / n_samples
            tesla_mean_psnr_value = tesla_running_psnr_value / n_samples
            
            dict_tesla_mean_metric_tb['test/SSIM'] = tesla_mean_ssim_value
            dict_tesla_mean_metric_tb['test/PSNR'] = tesla_mean_psnr_value
            
            # Visualize metric graphs
            for tag, value in dict_tesla_mean_metric_tb.items():
                
                print(f"{tag}: {value:.4f}")
                writer.scalar_summary(tag, value, step+1)  # Use .item() to get a Python number from a tensor containing a single value
            
            writer.writer.close()

            # (N/B, B, C, H, W) --> (B, C, H, W)
            data_A         = np.concatenate(data_A_list,         axis=0)
            data_PD        = np.concatenate(data_PD_list,        axis=0)
            data_B_4to1    = np.concatenate(data_B_4to1_list,    axis=0)
            data_B_2to1    = np.concatenate(data_B_2to1_list,    axis=0)
            data_B_4fold   = np.concatenate(data_B_4fold_list,   axis=0)
            data_B_2fold   = np.concatenate(data_B_2fold_list,   axis=0)
            data_B_HR      = np.concatenate(data_B_HR_list,      axis=0)
            data_B_SR_2to1 = np.concatenate(data_B_SR_2to1_list, axis=0)
            
            data_B_recon = np.concatenate(data_B_recon_list,  axis=0)
            c_b_sr_2to1_ds_1st   = np.concatenate(c_b_sr_2to1_ds_1st_list,    axis=0)
            c_b_sr_2to1_ds_2nd   = np.concatenate(c_b_sr_2to1_ds_2nd_list,    axis=0)
            c_b_sr_2to1_final    = np.concatenate(c_b_sr_2to1_final_list,     axis=0)

            #-------------------------------------------------------------------------------------------------------#
            # Data save
            data_save_dir = os.path.join(base_dir, f"output_data_1.2mm_{config.epochs}.h5")
            with h5py.File(data_save_dir, 'w') as f:

                # data_A
                f.create_dataset("data_A",    data=data_A)

                # data_PD
                f.create_dataset("data_PD",    data=data_PD)

                # data_B                
                f.create_dataset("data_B_HR",      data=data_B_HR)
                f.create_dataset("data_B_SR_2to1", data=data_B_SR_2to1)
                f.create_dataset("data_B_41",      data=data_B_4to1 )
                f.create_dataset("data_B_21",      data=data_B_2to1 )
                f.create_dataset("data_B_4fold",   data=data_B_4fold)
                f.create_dataset("data_B_2fold",   data=data_B_2fold)

                # data_A_recon
                f.create_dataset("data_B_recon", data=data_B_recon)
                
                # c_a
                f.create_dataset("data_c_b_ds_1st",  data=c_b_sr_2to1_ds_1st)
                f.create_dataset("data_c_b_ds_2nd",  data=c_b_sr_2to1_ds_2nd)
                f.create_dataset("data_c_B",       data=c_b_sr_2to1_final )

        log_command(config)        
        #-------------------------------------------------------------------------------------------------------#

    # cleanup()
#-----------------------------------------------------------------------------------------------------------

def str2bool(v):
    return v.lower() in ('true')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Through Plane Super Res')
    # current_date = datetime.date.today()

    # Training Configurations
    parser.add_argument('-gpu', '--gpu',         type=str,   default="7")
    parser.add_argument('-device', '--device',   type=str,   default="0")
    parser.add_argument('--dataset',             type=str,   default='IXI', choices=['IXI'])
    parser.add_argument('-w', '--workers',       type=int,   default=8,         help='number of data loader workers')
    parser.add_argument('-e', '--epochs',        type=int,   default=1,         help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size',    type=int,   default=10,        help='mini-batch size')
    parser.add_argument('-g_lr', '--gen_lr',     type=float, default=1e-4,      help='Learning rate of G')
    parser.add_argument('-d_lr', '--dis_lr',     type=float, default=1e-4,      help='Learning rate of D')
    parser.add_argument('-stacked_dsunet_lr', '--stacked_dsunet_lr', type=float, default=1e-4, help='Learning rate of Stacked_DSUNet')
    parser.add_argument('--beta1',               type=float, default=0.5,       help='beta1 for Adam optimizer')
    parser.add_argument('--beta2',               type=float, default=0.999,     help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay',        type=float, default=1e-4,      help='weight decay for Adam optimizer')
    parser.add_argument('--init',                type=str,   default='kaiming', help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr_policy',           type=str,   default='step',    help='learning rate scheduler')
    parser.add_argument('--step_size',           type=int,   default=20,        help='how often to decay learning rate when using step scheduler')
    parser.add_argument('--gamma',               type=float, default=0.1,       help='how much to decay learning rate when using StepLR scheduler')
    parser.add_argument('--patience',            type=int,   default=10,        help='how much to be patient for decaying learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--factor',              type=float, default=0.1,       help='how much to decay learning rate when using ReduceLROnPlateau scheduler')
    parser.add_argument('--gan_w',               type=int,   default=1,         help='weight of adversarial loss')
    parser.add_argument('--recon_x_w',           type=int,   default=10,        help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w',           type=int,   default=1,         help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w',           type=int,   default=1,         help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w',       type=int,   default=1,         help='weight of explicit style augmented cycle consistency loss')
    parser.add_argument('--recon_ssim_w',        type=int,   default=1,         help='weight of ssim loss')
    parser.add_argument('--recon_L1_w',          type=int,   default=1,         help='weight of l1 loss')
    parser.add_argument('--generator_steps',     type=int,   default=1,         help='number of steps for generator training')
    parser.add_argument('--discriminator_steps', type=int,   default=1,         help='number of steps for discriminator training')


    # Test configurations
    parser.add_argument('--test_epochs', type=int, default=10, help='test model from this epoch')

    # Data options
    parser.add_argument('--input_ch_a', type=int, default=1, help='number of image channels [1/3]')
    parser.add_argument('--input_ch_b', type=int, default=1, help='number of image channels [1/3]')
    
    parser.add_argument('-nb_train_imgs', '--nb_train_imgs', type=int, default=10, help='number of images employed for train code checking')
    parser.add_argument('-nb_test_imgs', '--nb_test_imgs',   type=int, default=10, help='number of images employed for test code checking')
    parser.add_argument('--train_dataset', type=str2bool, default=None,  help='Condition whether test the trained network on train dataset')


    # Model configurations
    # Generator
    parser.add_argument('--gen_dim', type=int, default=64, help='number of filters in the bottommost layer = the # of content code')
    parser.add_argument('--gen_mlp_dim', type=int, default=256, help='number of filters in MLP')
    parser.add_argument('--gen_style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--gen_activ', type=str, default='relu', help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--gen_n_downsample', type=int, default=2, help='number of downsampling layers in content encoder')
    parser.add_argument('--gen_n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')
    parser.add_argument('--gen_pad_type', type=str, default='reflect', help='padding type [zero/reflect]')

    # Discriminator
    parser.add_argument('--dis_dim', type=int, default=64, help='number of filters in the bottommost layer')
    parser.add_argument('--dis_norm', type=str, default='none', help='normalization layer [none/bn/in/ln]')
    parser.add_argument('--dis_activ', type=str, default='lrelu', help='activation function [relu/lrelu/prelu/selu/tanh]')
    parser.add_argument('--dis_n_layer',type=int, default=4, help='number of layers in D')
    parser.add_argument('--dis_gan_type', type=str, default='lsgan', help='GAN loss [lsgan/nsgan]')
    parser.add_argument('--dis_num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--dis_pad_type', type=str, default='reflect', help='padding type [zero/reflect]')

    # About PatchNCE loss
    parser.add_argument('--lambda_GAN',  type=float,    default=1.0, help='weight for GAN loss：GAN(G(X))')
    parser.add_argument('--lambda_NCE',  type=float,    default=1.0, help='weight for NCE loss: NCE(G(X), X)')
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
    # TESLA
    parser.add_argument('--log_dir',          type=str,  default='logs/tesla/')
    parser.add_argument('--ckpt_dir_tesla', type=str,  default='ckpts/tesla/', help='path to checkpoint directory')
    parser.add_argument('--pretrained_4to2',  type=bool, default=True, help='Options for whether to use pre-trained models')
    parser.add_argument('--pretrained_2to1',  type=bool, default=True, help='Options for whether to use pre-trained models')
    
    # ContentNet
    parser.add_argument('--ckpt_dir_contentNet',   type=str, default='ckpts/contentnet/', help='path to checkpoint directory')
    
    # Data dir
    parser.add_argument('--ixi_h5_1_2mm_dir', type=str, default='data/ixi/')
    
    # Miscellaneous
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--tb_display_size', type=int, default=3)
    parser.add_argument('--tb_comment', type=str, default='whatever_you_want/')   

    # Step size.
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--img_save_step', type=int, default=2)
    parser.add_argument('--img_display_step', type=int, default=2)
    parser.add_argument('--model_save_step', type=int, default=5)

    config = parser.parse_args()
    # print(config)
    main(config)