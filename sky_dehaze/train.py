import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from utils import logger, weight_init
from model import ACT
from data import HazeDataset
import torchvision.models as models
import math
import numpy as np


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),  # 将 4 通道图像转换为 3 通道
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg['ori_data_path'], cfg['haze_data_path'], data_transform)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                               num_workers=cfg['num_workers'], drop_last=True, pin_memory=True)

    val_haze_dataset = HazeDataset(cfg['val_ori_data_path'], cfg['val_haze_data_path'], data_transform)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
                                             num_workers=cfg['num_workers'], drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)    
    # data_transform = transforms.Compose([
    #     transforms.Resize([256, 256]),
    #     transforms.ToTensor()
    # ])
    # train_haze_dataset = HazeDataset(cfg['ori_data_path'], cfg['haze_data_path'], data_transform)
    # train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg['batch_size'], shuffle=True,
    #                                            num_workers=cfg['num_workers'], drop_last=True, pin_memory=True)

    # val_haze_dataset = HazeDataset(cfg['val_ori_data_path'], cfg['val_haze_data_path'], data_transform)
    # val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
    #                                          num_workers=cfg['num_workers'], drop_last=True, pin_memory=True)

    # return train_loader, len(train_loader), val_loader, len(val_loader)


@logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(path, net_name, '{}_{}.pkl'.format('', epoch)))


@logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    return optimizer


@logger
def load_network(device):
    net = ACT().to(device)
    net.apply(weight_init)
    return net


@logger
def loss_func(device):
    criterion = torch.nn.MSELoss().to(device)
    return criterion


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg['gpu'] > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # load data
    # print("00000000000000000000000000000000000000000000000")
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    # print("111111111111111111111111111111111111")
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    # -------------------------------------------------------------------
    # load network
    network = load_network(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    # start train

    print('Start train')
    network.train()
    for epoch in range(cfg['epochs']):
        Loss = 0
        for step, (ori_image, haze_image) in enumerate(train_loader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)
            Loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg['grad_clip_norm'])
            optimizer.step()

        print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.8f}'
              .format(epoch, cfg['epochs'], step + 1, train_number,
                      optimizer.param_groups[0]['lr'], Loss))
        # -------------------------------------------------------------------
        # start validation

        network.eval()
        Loss = 0
        for step, (ori_image, haze_image) in enumerate(val_loader):
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)

        print('VAL Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.4f}|PNSR: {: .4f}'
              .format(epoch + 1, cfg['epochs'], step + 1, train_number,
                      optimizer.param_groups[0]['lr'], loss.item(), 10 * math.log10(1.0 / loss.item())))

        torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),
                                                                 nrow=ori_image.shape[0]),
                                     os.path.join(cfg['sample_output_folder'], 'w{}_{}.jpg'.format(epoch, step)))

        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch + 1, cfg['model_dir'], network, optimizer, cfg['net_name'])
    # -------------------------------------------------------------------
    # train finish


if __name__ == '__main__':

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_args = {
        'epochs': 5,
        'lr': 1e-4,
        'use_gpu': True,
        'gpu': 0,
        'ori_data_path': os.path.join(current_dir, 'train_clear'),  # 训练集清晰图像路径
        'haze_data_path': os.path.join(current_dir, 'train_hazy'),  # 训练集雾霾图像路径
        'val_ori_data_path': os.path.join(current_dir, 'val_clear'),  # 验证集清晰图像路径
        'val_haze_data_path': os.path.join(current_dir, 'val_hazy'),  # 验证集雾霾图像路径
        'num_workers': 4,
        'batch_size': 30,
        'val_batch_size': 4,
        'print_gap': 500,
        'model_dir': os.path.join(current_dir, 'model'),  # 模型保存路径
        'log_dir': os.path.join(current_dir, 'model'),  # 日志保存路径
        'sample_output_folder': os.path.join(current_dir, 'samples'),  # 样本输出路径
        'net_name': 'dehaze_chromatic_',  # 网络名称
        'weight_decay': 0.0001,  # 权重衰减
        'grad_clip_norm': 1  # 梯度裁剪
    }

    main(config_args)
    
'''
python train.py --epochs 100 \
                --lr 1e-4 \
                --use_gpu true \
                --gpu 0 \
                --ori_data_path ./train_clear/ \
                --haze_data_path ./train_hazy \
                --val_ori_data_path ./val_clear/ \
                --val_haze_data_path ./val_hazy/ \
                --num_workers 4 \
                --batch_size 30 \
                --val_batch_size 4 \
                --print_gap 500 \
                --model_dir /model/ \
                --log_dir /model/ \
                --sample_output_folder /samples/ \
                --net_name /dehaze_chromatic_
'''