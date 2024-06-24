import numpy as np
import torch
from dataset import modeldata
from torch.utils.data import DataLoader
from PIL import Image   

colors = np.load("maskcolors.npy")

def save_image(model,device,epoch):
    test_data=modeldata("data")
    test_loader=DataLoader(test_data,batch_size=1,shuffle=True)
    model.eval()
    for idx,(data,mask) in enumerate(test_loader):
        data=data.to(device)
        copy=data[0]
        org_img=copy.cpu().numpy().astype(np.uint8)
        data=torch.movedim(data,-1,1)
        data=data.float()
        mask=mask.float()
        class_indices = np.argmax(mask[0], axis=-1)
        height, width = class_indices.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = colors[class_indices[i, j]]
        org_mask=image.astype(np.uint8)
        with torch.no_grad():
            output=model(data)
            output=torch.movedim(output,1,-1)
            output=torch.softmax(output,dim=-1)
            output=output.cpu().numpy()
            class_indices = np.argmax(output[0], axis=-1)
            height, width = class_indices.shape
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    image[i, j] = colors[class_indices[i, j]]
            gen_img =image.astype(np.uint8)
            
        conc_img=np.hstack((org_img,org_mask,gen_img))
        conc_img=Image.fromarray(conc_img)
        conc_img.save(f"result/{epoch}.png")
        break
    model.train()
    print("image saved")
    
def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
def check_accuracy(model,device):
    test_data=modeldata("test")
    test_loader=DataLoader(test_data,batch_size=1,shuffle=True)
    model.eval()
    with torch.no_grad():
        correct_pred=0
        total_pred=0
        for idx,(data,mask) in enumerate(test_loader):
            data=data.to(device)
            data=torch.movedim(data,-1,1)
            data=data.float()
            mask=mask.float()
            mask=mask.to(device)
            output=model(data)
            output=torch.movedim(output,1,-1)
            output=torch.softmax(output,dim=-1)
            _,predictions=torch.max(output,1)
            total_pred+=mask.numel()
            correct_pred+=(predictions==mask).sum()
        print(f"Accuracy: {(correct_pred/total_pred)}")
    model.train()