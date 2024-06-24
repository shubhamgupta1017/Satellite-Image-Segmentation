import torch
import torch.nn as nn
from dataset import modeldata
from model import unet
from tqdm import tqdm
import numpy as np
from utils import save_image,save_checkpoint,load_checkpoint,check_accuracy
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CHANNELS = 6
IN_CHANNELS = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "model.pth.tar"

def train_fn(train_loader, model, optimizer, loss_fn, scaler,epoch):
    loop=tqdm(train_loader,leave=True)
    for idx,(data,mask) in enumerate(loop):
        data=data.to(device).to(device)
        data=torch.movedim(data,-1,1)
        data=data.float()
        mask=mask.float().to(device)
        with torch.cuda.amp.autocast():
            output=model(data)
            output=torch.movedim(output,1,-1)
            loss=loss_fn(output,mask)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def main():
    model=unet(in_channels=IN_CHANNELS,out_channels=OUT_CHANNELS).to(device)
    model.train()
    train_dataset=modeldata("data")   
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=PIN_MEMORY,num_workers=NUM_WORKERS )
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    scaler=torch.cuda.amp.GradScaler()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_FILE),model,optimizer)
        
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scaler,epoch)
        save_image(model,device,epoch)
        
        if SAVE_MODEL and epoch%10==0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)
            check_accuracy(model,device)
if __name__=="__main__":
    main()
    