import os
import pdb
import h5py
import torch
import numpy as np
import random
from torch.utils import data
from PIL import Image
from monai.transforms import Transform, MapTransform, LoadImage, Compose, NormalizeIntensityd, Flipd, RandSpatialCrop, ResizeWithPadOrCropd, CropForeground, Resize, SpatialCrop, RandRotate
from monai.inferers.inferer import SlidingWindowInferer
from monai.config import IndexSelection, KeysCollection
from skimage.feature import canny
import pywt
    
class IXI_Dataset(data.Dataset):
    """Dataset class for the IXI dataset."""

    def __init__(self, config, mode):
        """Initialize and preprocess the IXI dataset."""

        # Train or test
        self.mode = mode

        self.config = config
        # Slice thickness of T2
        # self.slice_thickness = config.slice_thickness

        # Data directory
        self.ixi_h5_1_2mm_dir = config.ixi_h5_1_2mm_dir 

        if self.mode == "train":
            self.train_data_dir, self.nb_train_imgs = self.preprocess(config)
        elif self.mode == "test":
            self.test_data_dir, self.nb_test_imgs  = self.preprocess(config)

        self.data_dict = {}         

        # Transforms
        # self.resize = Resize(spatial_size=(128,256))

        
    def preprocess(self, config):
        # Set the # of imgs utilized for train and test
        
        if self.mode == 'train':                        
            data_dir = os.path.join(self.ixi_h5_1_2mm_dir, self.mode, f"output_data_1.2mm_2.4mm_4.8mm_50.h5")
            
            if config.nb_train_imgs is not None:
                assert isinstance(config.nb_train_imgs, int) and config.nb_train_imgs > 0, "config.nb_train_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_train_imgs <= len(f["data_B_HR"]), "config.nb_train_imgs should not exceed the total number of samples"
                nb_train_imgs = config.nb_train_imgs
            
            else: # config.nb_train_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 training 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_train_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_train_imgs
            
        elif self.mode == 'test':
            data_dir = os.path.join(self.ixi_h5_1_2mm_dir, self.mode, f"output_data_1.2mm_2.4mm_4.8mm_50.h5")
            
            if config.nb_test_imgs is not None:
                assert isinstance(config.nb_test_imgs, int) and config.nb_test_imgs > 0, "config.nb_test_imgs should be a positive interger"
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                # print(data_dir)
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_test_imgs <= len(f["data_B_HR"]), "config.nb_test_imgs should not exceed the total number of samples"
                nb_test_imgs = config.nb_test_imgs
            
            else: # config.nb_test_imgs에 아무 값도 주어지지 않았을때는 f["data_A"] 전체 slice 에 대해서 testing 하겠다는 뜻
                os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
                with h5py.File(data_dir, "r") as f:
                    nb_test_imgs = len(f["data_B_HR"])
                
            return data_dir, nb_test_imgs 
     
        
    def transforms(self, data):

        if self.mode == 'train':                
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"],subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
                # RandSpatialCrop(roi_size=(100,100),
                #                 random_size=False),
                # RandSpatialCrop(roi_size=(80,80),
                #                 random_size=False),
                # RandRotate( prob=0.1,
                #             range_x=[0.1, 0.1],
                #             range_y=[0.1, 0.1],
                #             range_z=[0.1, 0.1],
                #             mode='bilinear')        
            ])
            transformed_data = transforms( data )
            return transformed_data
        
        elif self.mode == 'test':
            transforms = Compose([
                # NormalizeIntensityd(keys=["A", "B"], subtrahend=None, divisor=None, nonzero=False, channel_wise=True),
                # ResizeWithPadOrCropd(keys=["A", "B"], spatial_size=(128, 256)),
                # Flipd(keys=["A","B"])
            ])
            transformed_data = transforms( data )
            return transformed_data
        

    def apply_canny(self, x_a):

        x_a_edge = []
            
        # Edge detection by using Canny filter
        x_a_edge_ = canny(x_a[0], sigma=1, low_threshold=0.1, high_threshold=0.2) # (128, 256)
        # np.expand_dims(x_a_edge, axis=0) # (1, 128, 256)
        x_a_edge.append(x_a_edge_)

        return x_a_edge

    def __getitem__(self, index):

        """
        __getitem__ 에서는 batch 차원은 없다고 생각하고 data 크기 따지는 것
        Dataloader에서 뱉어낼 때 batch 만큼 차원이 앞에 생기는 것

        Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """

        if self.mode == 'train':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.train_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 


            # Apply Canny filtering
            if self.config.crf_domain == "t1":
                cdt_edge = self.apply_canny(A)
            elif self.config.crf_domain == "pd":
                cdt_edge = self.apply_canny(PD)
            elif self.config.crf_domain == "t2":
                cdt_edge = self.apply_canny(B_HR)
            elif self.config.crf_domain == "srt2":    
                cdt_edge = self.apply_canny(B_SR_2to1)
            elif self.config.crf_domain == "none":
                cdt_edge = np.zeros_like(A)

            # Apply resizing        
            resize_2fold = Resize(spatial_size=(64,256), mode="area")
            resize_4fold = Resize(spatial_size=(32,256), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                         "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
                         "data_cdt_edge" : cdt_edge}
        
            self.data_dict.update(data_dict)
           
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"     ])),                                   
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
                                    "data_cdt_edge"         : torch.from_numpy(np.where(np.array(processed_data_dict["data_cdt_edge"]),1,0))
            }

            return processed_data_dict

        elif self.mode == 'test':

            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
            with h5py.File(self.test_data_dir, "r") as f:
                A         = np.copy( np.array(f["data_A"][index],         dtype=np.float32) )
                PD        = np.copy( np.array(f["data_PD"][index],        dtype=np.float32) )
                B_41      = np.copy( np.array(f["data_B_41"][index],      dtype=np.float32) )
                B_21      = np.copy( np.array(f["data_B_21"][index],      dtype=np.float32) )
                B_HR      = np.copy( np.array(f["data_B_HR"][index],      dtype=np.float32) )
                B_SR_4to2 = np.copy( np.array(f["data_B_SR_4to2"][index], dtype=np.float32) )
                B_SR_2to1 = np.copy( np.array(f["data_B_SR_2to1"][index], dtype=np.float32) ) 

            # Apply resizing        
            resize_2fold = Resize(spatial_size=(64,256), mode="area")
            resize_4fold = Resize(spatial_size=(32,256), mode="area")
            
            B_2fold = resize_2fold(B_HR)
            B_4fold = resize_4fold(B_HR)

            # Create dictionaries
            data_dict = {"data_A"        : A,
                         "data_PD"       : PD,
                         "data_B_HR"     : B_HR,
                         "data_B_41"     : B_41,
                         "data_B_21"     : B_21,
                         "data_B_4fold"  : B_4fold,
                         "data_B_2fold"  : B_2fold,
                         "data_B_SR_2to1": B_SR_2to1,
            }
            self.data_dict.update(data_dict)
            
            processed_data_dict = self.data_dict

            processed_data_dict = { "data_A"         : torch.from_numpy(np.array(processed_data_dict["data_A"   ])),
                                    "data_PD"        : torch.from_numpy(np.array(processed_data_dict["data_PD"  ])),
                                    "data_B_HR"      : torch.from_numpy(np.array(processed_data_dict["data_B_HR"])),
                                    "data_B_41"      : torch.from_numpy(np.array(processed_data_dict["data_B_41"])),
                                    "data_B_21"      : torch.from_numpy(np.array(processed_data_dict["data_B_21"])),
                                    "data_B_4fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_4fold"])),
                                    "data_B_2fold"   : torch.from_numpy(np.array(processed_data_dict["data_B_2fold"])),
                                    "data_B_SR_2to1" : torch.from_numpy(np.array(processed_data_dict["data_B_SR_2to1"])),
            }

            return processed_data_dict


    def __len__(self):
        """Returns the number of samples in the dataset."""

        if self.mode == 'train':
            # os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 HDF5 파일에서 동시에 읽고 쓰는 작업을 관리하기 위한 환경 변수 설정입니다.
            # HDF5 파일은 여러 프로세스 또는 스레드에서 동시에 접근할 수 있습니다.
            # 이때 HDF5_USE_FILE_LOCKING 환경 변수를 "TRUE"로 설정하면,
            # HDF5 라이브러리가 파일 잠금(file locking) 메커니즘을 사용하여 동시에 발생하는 접근 충돌을 방지합니다.
            # 따라서 os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"는 현재 스크립트에서 사용되는 HDF5 파일의 동시 접근 문제를 해결하기 위해 파일 잠금(file locking) 기능을 활성화하는 역할을 합니다.
            # Load
            return self.nb_train_imgs
        
        elif self.mode == 'test':

            return self.nb_test_imgs