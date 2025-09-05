import os
import os.path as osp
import torch
import random
import glob
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from measure_DIME import metrics_DIME
from torchinfo import summary
from eval import eval_func
from data.data import (
    get_lol_training_set, get_lol_v2_training_set,
    get_training_set_blur, get_lol_v2_syn_training_set,
    get_SID_training_set, get_SICE_training_set,
    get_SICE_eval_set, get_eval_set,
    get_fivek_training_set, get_fivek_eval_set,
    get_DIME_training_set, get_DIME_eval_set
)

from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from loguru import logger
import time
timestr = time.strftime('%Y%m%d%H%M%S')
LOGDIR = osp.join('logs', f'training')
if not osp.exists(LOGDIR):
    os.makedirs(LOGDIR)
logger.add(osp.join(LOGDIR, f'train_{timestr}.log'))

opt = option().parse_args()


def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    num_iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader, total=train_len):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)
            
        gt_rgb = im2  # original
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        num_iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        if num_iter == train_len:
            logger.info(f"===> Epoch[{epoch:06d}/{opt.nEpochs:06d}]: Loss: {(loss_last_10/pic_last_10):.6f} || Learning rate: lr={optimizer.param_groups[0]['lr']:.6e}.")
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.makedirs(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/output.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt


def checkpoint(epoch):
    checkpoint_dir = "./weights/train"
    if not os.path.exists(checkpoint_dir):          
        os.makedirs(checkpoint_dir)  
    
    # Save current checkpoint
    model_out_path = f"{checkpoint_dir}/epoch_{epoch:06d}.pth"
    torch.save(model.state_dict(), model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
    
    # Get all checkpoint files and sort by creation time
    checkpoint_pattern = os.path.join(checkpoint_dir, "epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files.sort(key=os.path.getctime)  # Sort by creation time
    
    # Keep only the 2 most recent checkpoints
    max_checkpoints = 2
    if len(checkpoint_files) > max_checkpoints:
        # Remove older checkpoints
        files_to_remove = checkpoint_files[:-max_checkpoints]
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                logger.info(f"Removed old checkpoint: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
    
    return model_out_path

def load_datasets():
    logger.info('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad or opt.fivek or opt.DIME:
        if opt.DIME:
            train_set = get_DIME_training_set(opt.data_train_DIME, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_DIME_eval_set(opt.data_val_DIME)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)            

        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_fivek_eval_set(opt.data_val_fivek)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    logger.info('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    '''
    preparation
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    logger.info(summary(model))
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.makedirs(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            elif opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            elif opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            elif opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            elif opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            elif opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            elif opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            elif opt.fivek:
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            elif opt.DIME:
                output_folder = 'DIME/'
                label_dir = opt.data_valgt_DIME  # shared_datasets/DIME/np/test/GT
                norm_size = False
            else:
                raise ValueError(f'Unknown dataset')
            # load `model_path` checkpoint and predict+visual results in testing_data_loader
            # saved to opt.val_folder + output_folder: `training/DIME/*.png`
            # f'{seq_name}_{osp.basename(lq_path).split(".")[0]}.png': 091_000.png
            logger.info(f"===> Epoch[{epoch:06d}/{opt.nEpochs:06d}] Start Evaluation")
            eval_func(model=model, testing_data_loader=testing_data_loader, model_path=model_out_path, output_folder=opt.val_folder + output_folder, norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            logger.info(f"===> Epoch[{epoch:06d}/{opt.nEpochs:06d}] End Evaluation")
            
            
            # im_dir = opt.val_folder + output_folder + '*.png'
            # if opt.DIME:
            #     # use_GT_mean: measured in gray
            #     avg_psnr, avg_ssim, avg_lpips = metrics_DIME(im_dir, label_dir, use_GT_mean=False)
            # else:
            #     avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            # logger.info(f"===> Avg.PSNR:  {avg_psnr:.6f} dB ")
            # logger.info(f"===> Avg.SSIM:  {avg_ssim:.6f} ")
            # logger.info(f"===> Avg.LPIPS:  {avg_lpips:.6f} ")
            # psnr.append(avg_psnr)
            # ssim.append(avg_ssim)
            # lpips.append(avg_lpips)
        torch.cuda.empty_cache()
    
    # now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # with open(f"./results/training/metrics{now}.md", "w") as f:
    #     f.write("dataset: "+ output_folder + "\n")  
    #     f.write(f"lr: {opt.lr}\n")  
    #     f.write(f"batch size: {opt.batchSize}\n")  
    #     f.write(f"crop size: {opt.cropSize}\n")  
    #     f.write(f"HVI_weight: {opt.HVI_weight}\n")  
    #     f.write(f"L1_weight: {opt.L1_weight}\n")  
    #     f.write(f"D_weight: {opt.D_weight}\n")  
    #     f.write(f"E_weight: {opt.E_weight}\n")  
    #     f.write(f"P_weight: {opt.P_weight}\n")  
    #     f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
    #     f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
    #     for i in range(len(psnr)):
    #         f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
        