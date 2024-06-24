from PIL import Image
import numpy as np
import os
from train import IMAGE_HEIGHT,IMAGE_WIDTH
files=os.listdir('Images/img')
class_colors=np.load('maskcolors.npy')
for idx,file in enumerate(files):
    img_path=os.path.join('Images/img',file)
    mask_path = os.path.join('Images/masks',file.replace('.jpg','.png'))
    try:
        mask_img=(Image.open(mask_path).convert("RGB"))
        img=(Image.open(img_path).convert("RGB"))
        mask_img=mask_img.resize((IMAGE_HEIGHT,IMAGE_WIDTH))
        img=img.resize((IMAGE_HEIGHT,IMAGE_WIDTH))
        mask_img=np.array(mask_img)
        img=np.array(img)
        h,w,_=mask_img.shape
        one_hot_encoded = np.zeros((h, w, len(class_colors)), dtype=np.uint8)
        for i, color in enumerate(class_colors):
            mask = np.all(mask_img == color, axis=-1)
            one_hot_encoded[mask, i] = 1
        np.save(f'data/mask/img_{idx}.npy', one_hot_encoded)
        np.save(f'data/image/img_{idx}.npy', img)
        if idx%50==0:
            print(f"Processed {idx} images")
            
    except:
        print(f"Error in {idx}")
        
num_images=len(os.listdir('data/image'))
num_masks=len(os.listdir('data/mask'))
print(f"done {num_images} images , {num_masks} masks and {len(files)}")