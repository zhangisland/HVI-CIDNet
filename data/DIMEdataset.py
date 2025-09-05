
import os
import os.path as osp
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import is_image_file, load_img
from torchvision import transforms as t


class DIMEDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        # data_dir: datasets/shared_datasets/DIME/np/test
        self.lq_data_info = []  # Store (path, sample)
        self.gt_data_info = []  # Store (path, sample)
        self.data_dir = data_dir
        self.transform = transform
        
        lq_folder = join(self.data_dir, 'LQ')
        lq_seqs = sorted([d for d in os.listdir(lq_folder) if os.path.isdir(join(lq_folder, d))])  # %03d, sequence name
        
        for seq in lq_seqs:
            seq_dir = join(lq_folder, seq)
            if os.path.exists(seq_dir):
                lq_files = [join(seq_dir, f) for f in os.listdir(seq_dir) if is_image_file(f)]
                lq_files.sort()
                # debug mode
                for f in lq_files[:3]:
                    gt_path = join(self.data_dir, 'GT', seq, os.path.basename(f))
                    self.lq_data_info.append((f, seq))
                    self.gt_data_info.append((gt_path, seq))

    def __getitem__(self, index):
        lq_path, seq_name = self.lq_data_info[index]
        gt_path, seq_name = self.gt_data_info[index]
        lq_input_img = load_img(lq_path)
        gt_input_img = load_img(gt_path)
        formatted_filename = f'{seq_name}_{osp.basename(lq_path).split(".")[0]}.png'
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            lq_input_img = self.transform(lq_input_img)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            gt_input_img = self.transform(gt_input_img)
        
        return lq_input_img, gt_input_img, formatted_filename, formatted_filename

    def __len__(self):
        return len(self.lq_data_info)

class DIMEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        # data_dir: datasets/shared_datasets/DIME/np/test
        self.lq_data_info = []  # Store (path, sample)
        
        self.data_dir = data_dir
        self.transform = transform
        
        lq_folder = self.data_dir
        lq_seqs = sorted([d for d in os.listdir(lq_folder) if os.path.isdir(join(lq_folder, d))])  # %03d, sequence name
        
        for seq in lq_seqs:
            seq_dir = join(lq_folder, seq)
            if os.path.exists(seq_dir):
                lq_files = [join(seq_dir, f) for f in os.listdir(seq_dir) if is_image_file(f)]
                lq_files.sort()
                for f in lq_files[:3]:
                    self.lq_data_info.append((f, seq))

    def __getitem__(self, index):
        lq_path, seq_name = self.lq_data_info[index]
        lq_input_img = load_img(lq_path)
        h, w = lq_input_img.size[:2]
        formatted_filename = f'{seq_name}_{osp.basename(lq_path).split(".")[0]}.png'
        
        if self.transform:
            lq_input_img = self.transform(lq_input_img)
        
        return lq_input_img, formatted_filename, h, w

    def __len__(self):
        return len(self.lq_data_info)


    

