import numpy as np
import os
from torch.utils.data import Dataset

class modeldata(Dataset):
    def __init__(self,dir):
        self.dir=dir
        self.image_dir=f"{self.dir}/image"
        self.mask_dir=f"{self.dir}/mask"
        self.images = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir,self.images[idx])
        mask_path= os.path.join(self.mask_dir,self.images[idx])
        img=np.load(img_path)
        mask=np.load(mask_path)
    
        return img,mask
        
if __name__=="__main__":
    dataset = modeldata()
    img,mask=dataset[0]
    print(img.shape,mask.shape)