from glob import glob
import os
import sys
import random

### USE:
### $python3 split.py DIR
### in some directory DIR with n class folders which all have data for that class
### this will arrange the data into training and validation for that particular directory
### e.g. DIR/class/1.jpg will be either in DIR/class/train/1.jpg or DIR/class/train/1.jpg


DATA_PATH = sys.argv[1]
proportion = 0.8
random.seed(11235812)


for path in glob(os.path.join(DATA_PATH, '*')):
    if not os.path.isdir(path):
        continue
    print(path)
    imgs = glob(os.path.join(path, '*.jpg'))
    imgs.sort()
    random.shuffle(imgs)
    try:
        os.mkdir(os.path.join(path, 'train'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(path, 'val'))
    except FileExistsError:
        pass
    split_idx = int(0.8 * len(imgs))
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]
    for img_path in train_imgs:
        target_path = os.path.join(path, 'train', img_path.split('/')[-1])
        os.rename(img_path, target_path)
        print(img_path, "  -->  ", target_path)
    for img_path in val_imgs:
        target_path = os.path.join(path, 'val', img_path.split('/')[-1])
        os.rename(img_path, target_path)
        print(img_path, "  -->  ", target_path)

    
