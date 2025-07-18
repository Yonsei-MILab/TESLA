# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import torch

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, config):
        """Initialize summary writer."""
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # tensorboard는 logdir 폴더 안에 있는 하위 폴더의 모든 event를 한 번에 보여주는 형식
        # config.log_dir -
        #                - self.current_time + config.tb_comment -
        #                                                        - event
        # config.log_dir 안에는 수많은 폴더들이 있고
        # 각 폴더안에는 event 들이 하나씩 생긴다
        # tensorboard 화면 좌측 Runs에서 보이는 이름들은
        # self.current_time + config.tb_comment --> 이 폴더명이다!
        # config.log_dir안에 training을 돌릴때마다 서로 다른 폴더를 만들어주기 위해서
        # self.current_time을 초단위까지 넣은 것
        # 그래야 training 돌릴때마다 각각의 tensorboard log를 만들고 기록할 수 있다.
        
        self.writer = SummaryWriter( logdir=config.log_dir + self.current_time + config.tb_comment ) 

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, global_step=step)

    def image_summary(self, tag, img_tensor, step):
        """Add image summary."""
        self.writer.add_image(tag, img_tensor, global_step=step)