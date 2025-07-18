import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision

from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric
from monai.transforms import Resize

def eval_net(trainer, eval_loader, config, epochs, device, writer ):
    
    torch.manual_seed(0)

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=config.data_range)
    psnr_metric = PSNRMetric(max_val=1)
    
    trainer.eval()
    n_val = len(eval_loader)  # the number of batch
    
    ProgDRL_running_ssim_value = 0
    ProgDRL_running_psnr_value = 0
    ProgDRL_mean_ssim_value = 0
    ProgDRL_mean_psnr_value = 0

    n_samples = 0

    dict_ProgDRL_mean_metric_tb = {}

    with tqdm(total=n_val, 
              desc='Validation', 
              unit='imgs', 
              ncols=180,
              ascii=' ==') as pbar:
        
        with torch.no_grad():
            assert config.tb_display_size <= config.batch_size, "tb_display_size는 batch_size보다 반드시 작거나 같아야합니다."
            """
            In distributed mode, calling the set_epoch() method at the beginning of each epoch
            before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
            Otherwise, the same ordering will be always used.        
            """
            # test_sampler.set_epoch( epochs )
                        
            for step, img in enumerate(eval_loader, len(eval_loader)):
                
                img["data_A"]         = img["data_A"].to( device )
                img["data_B_SR_2to1"] = img["data_B_SR_2to1"].to( device ) # super-resolved t2
                img["data_B_HR"]      = img["data_B_HR"].to( device )
                img["data_PD"]        = img["data_PD"].to( device )     

                pd_hr     = img["data_PD"]
                a_hr      = img["data_A"]
                b_sr_2to1 = img["data_B_SR_2to1"]
                b_hr      = img["data_B_HR"]

                trainer.eval()
                c_b_sr_2to1, s_b_sr_2to1_prime = trainer.gen_b.encode( b_sr_2to1 )
                x_b_sr_2to1_recon = trainer.gen_b.decode( c_b_sr_2to1[-1], s_b_sr_2to1_prime )
                trainer.train()

                # Compute ssim / psnr
                ssim_value = ssim_metric(y_pred=x_b_sr_2to1_recon, y=b_hr)
                psnr_value = psnr_metric(y_pred=x_b_sr_2to1_recon, y=b_hr)
                
                ProgDRL_running_ssim_value += torch.sum(ssim_value).item()
                ProgDRL_running_psnr_value += torch.sum(psnr_value).item()
                
                n_samples += b_hr.size(0)
                pbar.update(b_hr.shape[0])

                # =================================================================================== #
                #                                 3. Miscellaneous                                    #
                # =================================================================================== #

                # For ProgNet
                if config.hr_pd:
                    tb_a_hr           = torch.Tensor(pd_hr[:config.tb_display_size].cpu())    
                else:
                    tb_a_hr           = torch.Tensor(a_hr[:config.tb_display_size].cpu())
                tb_c_b_sr_2to1_ds_1st = torch.Tensor(c_b_sr_2to1[0][:config.tb_display_size, 0:1].cpu()) 
                tb_c_b_sr_2to1_ds_2nd = torch.Tensor(c_b_sr_2to1[1][:config.tb_display_size, 0:1].cpu()) 
                tb_c_b_sr_2to1_final  = torch.Tensor(c_b_sr_2to1[-1][:config.tb_display_size, 0:1].cpu())
                tb_b_sr_2to1          = torch.Tensor(b_sr_2to1[:config.tb_display_size].cpu())
                tb_recon_b_sr_2to1    = torch.Tensor(x_b_sr_2to1_recon[:config.tb_display_size].cpu()) # config.tb_display_size, 1, 128, 256
                tb_b_hr               = torch.Tensor(b_hr[:config.tb_display_size].cpu())              # config.tb_display_size, 1, 128, 256

                resize_transform = Resize(spatial_size=b_hr.shape[2:]) # Resize(spatial_size=(128,256))
            
                resize_c_b_sr_2to1_ds_1st = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_ds_1st[i][:]) for i in range(tb_c_b_sr_2to1_ds_1st.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_b_sr_2to1_ds_2nd = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_ds_2nd[i][:]) for i in range(tb_c_b_sr_2to1_ds_2nd.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                resize_c_b_sr_2to1_final  = torch.Tensor(torch.stack([resize_transform(tb_c_b_sr_2to1_final [i][:]) for i in range(tb_c_b_sr_2to1_final.shape[0])])) # tb에 그리기 위함이기 때문에 batch 만큼이 아니라 config.tb_display_size 만큼임
                
                ProgDRL_img_tensor = torch.cat([resize_c_b_sr_2to1_ds_1st,
                                                resize_c_b_sr_2to1_ds_2nd,
                                                resize_c_b_sr_2to1_final,
                                                tb_b_sr_2to1,
                                                tb_recon_b_sr_2to1,
                                                tb_b_hr,
                                                tb_a_hr] , dim=0)
                
                ProgDRL_img_grid = torchvision.utils.make_grid( tensor    = ProgDRL_img_tensor.data,
                                                                nrow      = config.tb_display_size, # 각 row 당 몇개의   이미지를 display 할건지
                                                                padding   = 0,
                                                                normalize = True )

            writer.image_summary( tag        = 'val/ProgDRL--cbDs1st--cbDs2nd--cbFinal--xb_sr--xbsrRecon--xb_hr--xa_hr',
                                  img_tensor = ProgDRL_img_grid,
                                  step       = epochs + 1 )
             
            ProgDRL_mean_ssim_value = ProgDRL_running_ssim_value / n_samples
            ProgDRL_mean_psnr_value = ProgDRL_running_psnr_value / n_samples

            dict_ProgDRL_mean_metric_tb['val/SSIM'] = ProgDRL_mean_ssim_value
            dict_ProgDRL_mean_metric_tb['val/PSNR'] = ProgDRL_mean_psnr_value

            # Visualize loss graphs
            for tag, value in dict_ProgDRL_mean_metric_tb.items():
                
                print(f"\n{tag}: {value:.4f}")
                writer.scalar_summary(tag, value, epochs+1)  # Use .item() to get a Python number from a tensor containing a single value
            