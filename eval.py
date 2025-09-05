import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet
from loguru import logger

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')
eval_parser.add_argument('--DIME', action='store_true', help='output DIME dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

ep = eval_parser.parse_args()

@torch.no_grad()
def eval_func(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    logger.info(f'Pre-trained model is loaded from {model_path}.')
    model.eval()
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                lq_input, name = batch[0], batch[1]
            else:
                lq_input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            lq_input = lq_input.cuda()
            output = model(lq_input**gamma) 
            
        if not os.path.exists(output_folder):          
            os.makedirs(output_folder)
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        for i in range(output.size(0)):
            img = output[i].detach().cpu()   # [C,H,W]

            img = img[[2,1,0], :, :]  # BGR->RGB

            output_img = transforms.ToPILImage()(img)
            output_img.save(os.path.join(output_folder, name[i]))
        
        # output_img = transforms.ToPILImage()(output.squeeze(0))
        # output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
        os.makedirs('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = CIDNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)

