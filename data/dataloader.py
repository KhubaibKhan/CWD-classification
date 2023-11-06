import torch.utils.data as data
from fmutils import fmutils as fmu

from PIL import Image
from pathlib import Path
import cv2, glob
import numpy as np
import os, random, time


from data.augmenters import data_augmenter
from data.utils import std_norm



# For CWD dataset
def cwd_splitter(main_dir, split):
    # split = ['train', 'val']
    folders = os.listdir(main_dir)
    img_dict = {}
    for folder in folders:
        img_dict[folder] = fmu.get_all_files(main_dir + folder)

    # split into train and test sets
    train_dict = {}
    test_dict = {}
    for key in img_dict.keys():
        random.shuffle(img_dict[key])
        train_dict[key] = img_dict[key][:int(len(img_dict[key])*split)]
        test_dict[key] = img_dict[key][int(len(img_dict[key])*split):]

    return train_dict, test_dict

class CWD_Dataset(data.Dataset):
    def __init__(self, data_dict, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        # convert keys into labels
        self.key2lbl = {self.keys[i]: i for i in range(len(self.keys))}
        self.lbl2key = {i: self.keys[i] for i in range(len(self.keys))}
        self.img_paths = []
        self.labels = []
        for key in self.keys:
            for img_path in self.data_dict[key]:
                self.img_paths.append(img_path)
                self.labels.append(self.key2lbl[key])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = Image.open(img_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[index]

def cwd_loader(data_dict, transform=None, batch_size=32, shuffle=True, num_workers=4):
    # data_dict = {'class1': [img1, img2, img3, ...], 'class2': [img1, img2, img3, ...]}
    dataset = CWD_Dataset(data_dict, transform=transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader



# For iNat dataset
def inat_files(main_dir):

    tv_files = []
    split = ['train_200k', 'val']
    for s in split:
        dirs = fmu.get_all_dirs(main_dir + s)
        plant_dirs = [Path(dirs[i]).stem for i in range(len(dirs)) if 'Plantae' in dirs[i]]
        dir_names = sorted(plant_dirs)
        if s == 'train_200k':
            dir_to_label = {dir_name: i for i, dir_name in enumerate(dir_names)}

        plant_dirs = [dirs[i] for i in range(len(dirs)) if 'Plantae' in dirs[i]]
        # get all files from the dirs
        all_files = []
        for dir in plant_dirs:
            path_pattern = os.path.join(dir, '*')
            files = glob.glob(path_pattern)
            all_files.extend(files)

        tv_files.append(all_files)
    return tv_files[0], tv_files[1], dir_to_label

class iNatDataset(data.Dataset):
    def __init__(self, img_paths, dir2lbl, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.key2lbl = dir2lbl
        self.lbl2key = {v: k for k, v in self.key2lbl.items()}
        self.filenames = img_paths

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        
        idx = self.key2lbl[Path(self.img_paths[index]).parent.name]

        if self.transform:
            img = self.transform(img)

        return img, idx
    

def inat_loader(files, dir2lbl, transform=None, batch_size=32, shuffle=True, num_workers=4):
    # data_dict = {'class1': [img1, img2, img3, ...], 'class2': [img1, img2, img3, ...]}
    dataset = iNatDataset(files, dir2lbl=dir2lbl, transform=transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader