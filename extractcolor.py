import numpy as np
import os
from PIL import Image
maskcol=[]
if __name__=="__main__":
    files=os.listdir("Images/masks")
    for idx,i in enumerate(files):
        try:
            mask=np.array(Image.open(f"Images/masks/{i}").convert("RGB"))
            pixels = mask.reshape((-1, 3))
            unique_pixels = np.unique(pixels, axis=0)
            for color in unique_pixels:
                if tuple(color) == (0,0,0):
                    print(f"skipped {color} {i}")
                if tuple(color) not in maskcol:
                    maskcol.append(tuple(color))
                    print(f"added {color} {i}")
            print(f"done {i}")
        except:
            print(f"failed{i}")
    colorarray=np.array(maskcol)
    print(colorarray)
    np.save("maskcolors.npy",colorarray)