import os
import tqdm
import configparser
import json
import warnings
from PIL import Image
import matplotlib.pyplot as plt

import wandb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

import losses
import utils


class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=3, warmup_lr_init=1e-5, 
                 min_lr=1e-5,
                 last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
            
        if self.last_epoch == 0:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]

        if self.last_epoch < self.warmup_epochs:
            total_steps = self.warmup_epochs
            steps_num = self.last_epoch
            
            delta_step = (self.base_lrs[-1] - self.warmup_lr_init) / total_steps
            for group in self.optimizer.param_groups:
                group['lr'] = steps_num * delta_step + self.warmup_lr_init
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if (self.last_epoch - self.warmup_epochs) % self.step_size:
            return [group['lr'] for group in self.optimizer.param_groups]

        else:
            candidadates_lr = self.optimizer.param_groups[-1]['lr'] * self.gamma
            if candidadates_lr >= self.min_lr:
                self.optimizer.param_groups[-1]['lr'] = self.optimizer.param_groups[-1]['lr'] * self.gamma
                return [group['lr'] for group in self.optimizer.param_groups]

        return self.get_last_lr()

def get_scheduler(opt, config):
    step_size = json.loads(config.get("training", "step_size"))
    gamma = json.loads(config.get("training", "gamma"))
    warmup_epochs = json.loads(config.get("training", "warmup_epochs"))
    warmup_lr_init = json.loads(config.get("training", "warmup_lr_init"))
    min_lr = json.loads(config.get("training", "min_lr"))
    
    scheduler = StepLRWithWarmup(opt, step_size=step_size, gamma=gamma, 
                                      warmup_epochs=warmup_epochs, warmup_lr_init=warmup_lr_init,
                                      min_lr=min_lr)
    return scheduler

@torch.no_grad()
def evaluate(model, test_dataloader, device):
    n_classes = 5
    metric_mIOS = losses.mIOS(n_classes=n_classes).to(device)
    mIOS_mean = 0
    accuracy = 0
    model.eval()
    for img, mask in test_dataloader:
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        pred_class = torch.argmax(pred, dim=1)
        accuracy += (pred_class == mask).float().mean() 
        mIOS_mean += metric_mIOS(pred, mask)
    ios_class = metric_mIOS.ios_class.cpu().numpy() / len(test_dataloader)
    result = {}
    for class_ in range(n_classes):
        result[f"IOS {utils.classes[class_]}"] = ios_class[class_]
    result["accuracy"] = accuracy.cpu().item() / len(test_dataloader)
    result["mean IOS"] = ios_class.mean()
    return result
    
    
def train_epoch(model, loss, opt, train_dataloader, device):
    epoch_loss = 0
    model.train()
    for img, mask in train_dataloader:
        opt.zero_grad()
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss_t = loss(pred, mask)
        loss_t.backward()
        opt.step()
        epoch_loss += loss_t
    epoch_loss = epoch_loss / len(train_dataloader.dataset)
    return epoch_loss


def train_epoch_weighted(model, loss, opt, train_dataloader, device):
    epoch_loss = 0
    model.train()
    for img, mask, weights in train_dataloader:
        opt.zero_grad()
        img = img.to(device)
        weights = weights.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss_t = loss(pred, mask, weights)
        loss_t.backward()
        opt.step()
        epoch_loss += loss_t
    epoch_loss = epoch_loss / len(train_dataloader.dataset)
    return epoch_loss

def train_epoch_common_masked(model, loss, opt, train_dataloader, device):
    epoch_loss = 0
    model.train()
    model.to(device)
    for batch in train_dataloader:
        msc = batch["msc"]
        img_msc = msc[0].to(device)
        mask = msc[1].to(device)
        pred_usa_encoder = model.encoder(batch["usa"].to(device))
        pred_msc_encoder = model.encoder(img_msc)
        pred_msc = model.segmentation_head(model.decoder(*pred_msc_encoder))
        
        opt.zero_grad()
        loss_t = loss(pred_msc_encoder, mask, pred_msc, pred_usa_encoder)
        loss_t.backward()
        opt.step()
        
        epoch_loss += loss_t
    epoch_loss = epoch_loss / len(train_dataloader.dataset)
    return epoch_loss

def log(logs, epoch):
    wandb.log(data = logs, step=epoch)

def train_main_model(model, loss, opt, scheduler,
                     train_dataloader, test_dataloader_1, test_dataloader_2, name, epoch_t=100, device=torch.device("cpu"),
                     log_flag=True, train_area="msc", test_area_1="msc", test_area_2="usa"):
    path = f"./saved_data/{name}"
    if not os.path.isdir(path):
        os.mkdir(path)
    model = model.to(device)
    loss = loss.to(device)
    logs = {}
    for epoch in tqdm.tqdm(range(epoch_t)):
        logs[f"epoch loss {train_area}"] = train_epoch_common_masked(model, loss, opt, train_dataloader, device).cpu().item()
        scheduler.step()
        
        eval_logs = evaluate(model, test_dataloader_1, device)
        for key, value in eval_logs.items():
            logs[f"{key} {test_area_1}"] = value
            
        eval_logs = evaluate(model, test_dataloader_2, device)
        for key, value in eval_logs.items():
            logs[f"{key} {test_area_2}"] = value
            
        if not epoch % 10:
            torch.save(model.state_dict(), f"saved_data/{name}/epoch_{epoch}")
        logs["last lr"] = scheduler.get_last_lr()[-1]
        if log_flag:
            log(logs, epoch)
    torch.save(model.state_dict(), f"saved_data/{name}/final")
    return model

def train_classifier_model(model, loss, opt, scheduler,
                     train_dataloader, epoch_t=100, device=torch.device("cpu"), log_flag=True):
    model = model.to(device)
    loss = loss.to(device)
    logs = {}
    for epoch in tqdm.tqdm(range(epoch_t)):
        logs[f"epoch loss classification"] = train_epoch(model, loss, opt, train_dataloader, device).cpu().item()
        scheduler.step()
        logs["last lr"] = scheduler.get_last_lr()[-1]
        if log_flag:
            log(logs, epoch)
    return model

@torch.no_grad()
def get_predicted_cls(model, dataloader, device, temperature=10):
    temperature = torch.tensor(temperature).to(device)
    model.eval()
    result = {}
    preds = []
    pathes = []
    for data in dataloader:
        imges = data["img"].to(device)
        preds = nn.functional.softmax(model(imges) / temperature, dim=1)[:, 1].cpu()
        pathes = data["path"]
        result.update({path: pred for path, pred in zip(pathes, preds)})
    return result

def get_img_pred_mask(model, dataloader, idx):
    pal = [value for color in utils.PALLETE for value in color]
    img, mask = dataloader.dataset[idx]
    pred = torch.argmax(model(img[None]), dim=1).squeeze().numpy().astype(np.uint8)
    label = mask.squeeze().numpy().astype(np.uint8)
                                         
    img_path = dataloader.dataset.img_path
    file_name = dataloader.dataset.file_names[idx]
    path = img_path / (str(file_name) + ".tif")
    _, img = utils.convert(path)

    label = Image.fromarray(label).convert('P')
    label.putpalette(pal)

    pred = Image.fromarray(pred).convert('P')
    pred.putpalette(pal)
    
    return img, pred, label
    
@torch.no_grad()
def cls_sainty_check(cls_model, dataloader, device, label):
    cls_model.eval()
    cls_model = cls_model.to(device)
    true_answers = 0
    for batch in dataloader:
        images = batch[0].to(device)
        labels = label * torch.ones(images.shape[0]).to(device)
        preds = torch.argmax(cls_model(images), dim=1)
        true_answer = (preds == labels).float().mean()
        true_answers += true_answer
    true_answers = true_answers / len(dataloader)
    return true_answers.cpu().item()

@torch.no_grad()
def sainty_check(model, test_dataloader_usa, test_dataloader_msc, device, idxs=(0, 0)):
    model = model.to(device)
    eval_logs_usa = evaluate(model, test_dataloader_usa, device)
    eval_logs_msc = evaluate(model, test_dataloader_msc, device)
    model = model.to("cpu")

    plt.figure(figsize=(12, 12))
    
    img_usa, pred_usa, label_usa = get_img_pred_mask(model, test_dataloader_usa, idxs[0])
    img_msc, pred_msc, label_msc = get_img_pred_mask(model, test_dataloader_msc, idxs[1])
    
    
    plt.subplot(2, 3, 1)
    plt.title("MSC")
    plt.imshow(img_msc)
    
    plt.subplot(2, 3, 2)
    plt.title("label MSC")
    plt.imshow(label_msc)

    plt.subplot(2, 3, 3)
    plt.title("pred MSC")
    plt.imshow(pred_msc)

    plt.subplot(2, 3, 4)
    plt.title("USA")
    plt.imshow(img_usa)
    
    plt.subplot(2, 3, 5)
    plt.title("label USA")
    plt.imshow(label_usa)

    plt.subplot(2, 3, 6)
    plt.title("pred USA")
    plt.imshow(pred_usa)

    plt.savefig("results.eps")
    plt.show();
    
    return {"msc": eval_logs_msc, "USA": eval_logs_usa}
    
    
    