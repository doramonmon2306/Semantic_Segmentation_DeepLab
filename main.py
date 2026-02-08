import os
import torch
import numpy as np
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
import shutil
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

class VOCDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transform, target_transform):
        super().__init__(root, year, image_set, download, transform, target_transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        
    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        label = np.array(label, np.int64)
        label[label == 255] = 0
        return image, label
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #------------------------------------------------------------------
    path = "VOC" #Download the dataset and replace your path here
    #------------------------------------------------------------------
    transform = Compose([
        Resize((224,224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = Resize((224,224))
    train_dataset = VOCDataset(root=path, year="2012", image_set="train", download=False,
                               transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_dataset = VOCDataset(root=path, year="2012", image_set="val", download=False,
                             transform=transform, target_transform=target_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model = model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    acc_metric = MulticlassAccuracy(num_classes=21).to(device)
    mIOU_metric = MulticlassJaccardIndex(num_classes=21).to(device)
    bestmIOU = -1
    num_iter_per_epoch = len(train_dataloader)
    tb = "tensorboard"
    if os.path.isdir(tb):
        shutil.rmtree(tb)
    writer = SummaryWriter(tb)
    
    for epoch in range(100):
        #TRAINING
        model.train()
        progress_bar = tqdm.tqdm(train_dataloader, colour = "cyan")
        losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.long().to(device)  
            output = model(images)
            predict = output["out"]
            loss_value = criterion(predict, labels)
            losses.append(loss_value.cpu().item())
            loss_avg = np.mean(losses)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch+1, 100, loss_avg))
            writer.add_scalar("Train/Loss", loss_avg, epoch*num_iter_per_epoch + iter)
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        #VALIDATION
        model.eval()
        progress_bar = tqdm.tqdm(val_dataloader, colour="yellow")
        acc_metric.reset()
        mIOU_metric.reset()
        
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device) 
            labels = labels.long().to(device)  
            with torch.no_grad():  
                output = model(images)['out']
            acc_metric.update(output, labels)
            mIOU_metric.update(output, labels)
                        
        acc = acc_metric.compute().item()
        mIOU = mIOU_metric.compute().item()
        writer.add_scalar("Test/accuracy", acc, epoch)
        writer.add_scalar("Test/mIOU", mIOU, epoch)
        scheduler.step(mIOU)
            
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": acc,
            "mIOU": mIOU
            }
            
        if mIOU > bestmIOU:
            bestmIOU = mIOU
            torch.save(checkpoint, "best_model.pth")
            print(f"Saved best model with accuracy: {acc:.4f}, mIOU: {mIOU:.4f}")
            
            
if __name__ == '__main__':
    train()