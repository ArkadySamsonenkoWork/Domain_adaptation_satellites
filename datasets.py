import os
from pathlib import Path
import random
import configparser
import json

import wandb

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms

import tifffile

class DoubleTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        
    def forward(self, img, mask):
        if random.random() < 0.5:
            return self.transform(img), self.transform(mask)
        else:
            return img, mask
    
class HFlip(DoubleTransform):
    def __init__(self):
        transform = transforms.functional.hflip
        super().__init__(transform)
        
class VFlip(DoubleTransform):
    def __init__(self):
        transform = transforms.functional.vflip
        super().__init__(transform)

class SingleTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        
    def forward(self, img, mask):
        return self.transform(img), mask

class ColorJitter(nn.Module):
    def __init__(self, coeff):
        self.coeff = coeff
        super().__init__()
        self.generator = torch.rand
        
    def forward(self, input):
        chanels = input.shape[0]
        std = input.std(dim=(1, 2))
        shift = self.coeff * (2 * self.generator(chanels) - 1) * std
        #contr = self.coeff * (2 * self.generator(chanels) - 1)
        #return (input + shift[:, None, None]) * contr[:, None, None]
        return input + shift[:, None, None]

    
class SingleColorJitter(SingleTransform):
    def __init__(self):
        transform = ColorJitter(0.05)
        super().__init__(transform)
        
class LandcoverDataset(Dataset):
    def __init__(self, img_path, mask_path, file_names, size=256, resize_mask=True, aug=None, transform=None):
        self.resize = transforms.Resize(size)
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = file_names
        self.size = size
        self.aug = aug
        self.images = []
        self.masks = []
        for name in self.file_names:
            image = self.get_image(name, transform)
            self.images.append(image)
            mask = self.get_mask(name, resize_mask)
            self.masks.append(mask)

    def get_image(self, name, transform):
        path = self.img_path / (name.name + ".tif")
        image = tifffile.imread(path)
        #image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_CUBIC)  
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        
        image = image.astype(np.float32, copy=False)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.resize(image)
        if transforms is not None:
            image = transform(image)
        return image
            
    def get_mask(self, name, resize_mask):
        path = self.mask_path / (name.name + ".tif")
        mask = tifffile.imread(path)
        mask = torch.tensor(mask, dtype=torch.long)
        mask = self.resize(mask[None, ...]).squeeze()
        return mask
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.aug is not None:
            for aug in self.aug:
                image, mask = aug(image, mask)
        return image, mask


class CommonMaskedDataset(Dataset):
    def __init__(self):
        img_path_rus, mask_path_rus, train_data_rus, _ = get_msc_split()
        img_path_usa, mask_path_usa, train_data_usa, _ = get_usa_split()

        mean, std = get_mean_std()
        
        transform = transforms.Normalize(mean=mean, std=std)
        train_aug = [
            HFlip(),
            VFlip(),
        ]
        self.msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
        self.usa_dataset = LandcoverDataset(img_path_usa, mask_path_usa, train_data_usa,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)

    def __len__(self):
        return len(self.msc_dataset)

    def __getitem__(self, idx):
        msc_len = len(self.msc_dataset)
        usa_len = len(self.usa_dataset)
        idx_usa = random.randint(0, usa_len-1)
        
        return {"msc": self.msc_dataset[idx], "usa": self.usa_dataset[idx_usa][0]}


def get_common_masked_dataloader(batch_size=32):
    common_dataset = CommonMaskedDataset()
    train_dataloader = DataLoader(common_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    return train_dataloader

    
class WeightedLandcoverDataset(LandcoverDataset):
    def __init__(self, img_path, mask_path, file_names, weights, size=256, resize_mask=True, aug=None, transform=None):
        super().__init__(img_path, mask_path, file_names, size, resize_mask, aug, transform)
        self.weights = [weights[file_name.name] for file_name in self.file_names]

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        return image, mask, self.weights[idx]

def get_weighted_dataloader(weights):
    batch_size =32
    names = weights.keys()
    names = [Path(name) for name in names]
    img_path = Path("./datasets/datasets/Russia/train/images/")
    mask_path = Path("./datasets/datasets/Russia/train/masks/")
    mean, std = get_mean_std()
        
    transform = transforms.Normalize(mean=mean, std=std)
    train_aug = [
            HFlip(),
            VFlip(),
    ]
    dataset = WeightedLandcoverDataset(img_path, mask_path, names, weights, resize_mask=True, aug=train_aug, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    return train_dataloader

def get_part_dataloader(weights, top_p=25):
    part = 1 - top_p / 100
    batch_size = 32
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}
    weights = dict(list(weights.items())[int(len(weights) * part):])
    names = weights.keys()
    names = [Path(name) for name in names]
    img_path = Path("./datasets/datasets/Russia/train/images/")
    mask_path = Path("./datasets/datasets/Russia/train/masks/")
    mean, std = get_mean_std()
        
    transform = transforms.Normalize(mean=mean, std=std)
    train_aug = [
            HFlip(),
            VFlip(),
    ]
    dataset = LandcoverDataset(img_path, mask_path, names, resize_mask=True, aug=train_aug, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    return train_dataloader
    

class CommonDataset(Dataset):
    def __init__(self):
        img_path_rus, mask_path_rus, train_data_rus, _ = get_msc_split()
        img_path_usa, mask_path_usa, train_data_usa, _ = get_usa_split()

        mean, std = get_mean_std()
        
        transform = transforms.Normalize(mean=mean, std=std)
        train_aug = [
            HFlip(),
            VFlip(),
        ]
        self.msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
        self.usa_dataset = LandcoverDataset(img_path_usa, mask_path_usa, train_data_usa,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)

    def __len__(self):
        return len(self.msc_dataset) + len(self.usa_dataset)

    def __getitem__(self, idx):
        msc_len = len(self.msc_dataset)
        if idx >= msc_len:
            idx = idx - msc_len
            return self.usa_dataset[idx]
        else:
            return self.msc_dataset[idx]


class CommonDatasetBare(Dataset):
    def __init__(self):
        img_path_rus, mask_path_rus, train_data_rus, _ = get_msc_split()
        img_path_usa, mask_path_usa, train_data_usa, _ = get_usa_split()

        mean, std = get_mean_std()
        
        transform = transforms.Normalize(mean=mean, std=std)
        train_aug = [
            HFlip(),
            VFlip(),
        ]

        train_data_rus = self.resample_data_bare(img_path_rus, mask_path_rus, train_data_rus)
        
        self.msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
        self.usa_dataset = LandcoverDataset(img_path_usa, mask_path_usa, train_data_usa,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)

    
    def resample_data_bare(self, img_path_rus, mask_path_rus, train_data_rus, top_k=0.2):
        mean, std = get_mean_std()

        transform = transforms.Normalize(mean=mean, std=std)
        msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=None, transform=transform)
        bare_concs = [] 
        bare_cls_idx = 2
        size = len(msc_dataset)
        for mask in msc_dataset.masks:
            bare_amount = (mask==bare_cls_idx).sum()
            bare_conc = bare_amount / mask.numel()
            bare_concs.append(bare_conc.item())
        indexes = np.argsort(bare_concs)[int((1-top_k) * size):]
        train_data_rus = [train_data_rus[idx] for idx in indexes]
        return train_data_rus
        
            
    
    def __len__(self):
        return len(self.msc_dataset) + len(self.usa_dataset)

    def __getitem__(self, idx):
        msc_len = len(self.msc_dataset)
        if idx >= msc_len:
            idx = idx - msc_len
            return self.usa_dataset[idx]
        else:
            return self.msc_dataset[idx]
        
def get_dataloaders_common(batch_size=32):
    common_dataset = CommonDataset()
    train_dataloader = DataLoader(common_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    return train_dataloader

def get_dataloaders_bare(batch_size=32):
    common_dataset = CommonDatasetBare()
    train_dataloader = DataLoader(common_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    return train_dataloader
    
class BinaryDataset(Dataset):
    def __init__(self, img_path_rus, mask_path_rus, train_data_rus,
                 img_path_usa, mask_path_usa, train_data_usa, train_aug, transform
                ):
        
        self.msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
        self.usa_dataset = LandcoverDataset(img_path_usa, mask_path_usa, train_data_usa,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
    def __len__(self):
        return len(self.msc_dataset) + len(self.usa_dataset)

    def __getitem__(self, idx):
        msc_len = len(self.msc_dataset)
        if idx >= msc_len:
            idx = idx - msc_len
            return self.usa_dataset[idx][0], torch.tensor(1)
        else:
            return self.msc_dataset[idx][0], torch.tensor(0)
        
class PathImageDataset(Dataset):
    def __init__(self, img_path_rus, mask_path_rus, train_data_rus, train_aug, transform):
        self.msc_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data_rus,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
        self.pathes = train_data_rus

    def __len__(self):
        return len(self.msc_dataset)

    def __getitem__(self, idx):
        return {"img": self.msc_dataset[idx][0], "path": str(self.pathes[idx])}

class DatasetsSamplingCreator():
    def __init__(self, splits=3):
        img_path_rus, mask_path_rus, train_data_rus, _ = get_msc_split()
        img_path_usa, mask_path_usa, train_data_usa, _ = get_usa_split()
        
        mean, std = get_mean_std()
        self.msc_pathes = [img_path_rus, mask_path_rus, train_data_rus]
        self.usa_pathes = [img_path_usa, mask_path_usa, train_data_usa]
        
        self.transform = transforms.Normalize(mean=mean, std=std)
        self.train_aug = [
            HFlip(),
            VFlip(),
        ]
        self.splits = splits
    
    def __getitem__(self, idx):
        batch_size = 32
        img_path_rus, mask_path_rus, train_data_rus = self.msc_pathes
        img_path_usa, mask_path_usa, train_data_usa = self.usa_pathes
        idx_batch = len(train_data_rus) // self.splits
        if idx >= self.splits:
            raise IndexError
        train_data_rus_val = train_data_rus[idx_batch * idx: idx_batch * (idx + 1)]
        #print(f"val {train_data_rus_val}")
        if idx==0:
            train_data_rus_train = train_data_rus[:idx_batch * idx] + train_data_rus[idx_batch * (idx + 1):]
        else:
            train_data_rus_train = train_data_rus[idx_batch * (idx - 1):idx_batch * idx] + train_data_rus[idx_batch * (idx + 1):]
    
        binary_dataset = BinaryDataset(img_path_rus, mask_path_rus, train_data_rus_train, 
                                       img_path_usa, mask_path_usa, train_data_usa, self.train_aug, self.transform)
        binary_dataloader = DataLoader(binary_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

        path_dataset = PathImageDataset(img_path_rus, mask_path_rus, train_data_rus_val, self.train_aug, self.transform)
        path_dataloader = DataLoader(path_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
        return binary_dataloader, path_dataloader
        
def get_mean_std():
    config = configparser.ConfigParser()
    config.read("config.ini")
    mean = json.loads(config.get("dataset", "mean"))
    std = json.loads(config.get("dataset", "std"))
    return mean, std

def get_data_list(img_path):
    """
    Retrieves a list of file names from the given directory.
    """
    name = []
    for _, _, filenames in os.walk(img_path):  # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            name.append(f)
    return name

def get_msc_split():
    check_size = 512
    data_list_rus = get_data_list("./datasets/datasets/Russia/train/images")
    data_list_msc = [Path(name) for name in data_list_rus if name.startswith("msc")]
    img_path_rus = Path("./datasets/datasets/Russia/train/images/")
    mask_path_rus = Path("./datasets/datasets/Russia/train/masks/")
    data_list_train = []
    for name in data_list_msc:
        path = img_path_rus / (name.name + ".tif")
        image = tifffile.imread(path)
        if (image.shape[0] == check_size) and (image.shape[1] == check_size):
            data_list_train.append(name)

    random.seed(4)
    random.shuffle(data_list_train)
    train_data = data_list_train[:4000]
    test_data = data_list_train[4000:]
    return img_path_rus, mask_path_rus, train_data, test_data

def get_usa_split():
    data_list_usa = [Path(img_path) for img_path in get_data_list("./datasets/datasets/USA/train/images")]
    img_path_usa = Path("./datasets/datasets/USA/train/images/")
    mask_path_usa = Path("./datasets/datasets/USA/train/masks/")
    
    random.seed(4)
    random.shuffle(data_list_usa)
    train_data = data_list_usa[:1000]
    test_data = data_list_usa[1000:1500]
    return img_path_usa, mask_path_usa, train_data, test_data

def get_dataloaders_msc(batch_size):
    img_path_rus, mask_path_rus, train_data, test_data = get_msc_split()
    mean, std = get_mean_std()
    
    transform = transforms.Normalize(mean=mean, std=std)
    train_aug = [
        HFlip(),
        VFlip(),
        # SingleColorJitter(),
    ]
    trian_dataset = LandcoverDataset(img_path_rus, mask_path_rus, train_data,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
    test_dataset = LandcoverDataset(img_path_rus, mask_path_rus, test_data,
                                    size=256, resize_mask=True, aug=None, transform=transform)
    
    train_dataloader = DataLoader(trian_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    return train_dataloader, valid_dataloader
    
    
def get_dataloaders_usa(batch_size):
    img_path_usa, mask_path_usa, train_data, test_data = get_usa_split()
    mean, std = get_mean_std()
    transform = transforms.Normalize(mean=mean, std=std)
    train_aug = [
        HFlip(),
        VFlip(),
        # SingleColorJitter(),
    ]
    trian_dataset = LandcoverDataset(img_path_usa, mask_path_usa, train_data,
                                     size=256, resize_mask=True, aug=train_aug, transform=transform)
    test_dataset = LandcoverDataset(img_path_usa, mask_path_usa, test_data,
                                    size=256, resize_mask=True, aug=None, transform=transform)
    
    train_dataloader = DataLoader(trian_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    return train_dataloader, valid_dataloader