import numpy as np
import random
import os
from glob import glob
from scipy import misc
from skimage.transform import resize

class ImageDataset(object):
    #Image dataset using skimage/numpy, slower than tf data api, which is a bit more complicated but faster
    #Represents a dataset and should do all the handling of datasets 

    def __init__(self, data_dir, h, w, batch_size, crop_proportion, glob_pattern='*/*.jpg'):
        self.data_dir = data_dir
        print(self.data_dir)
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.idx = 0
        #Crop proportion is approximately how much of the image we retain when we crop
        #e.g. .9 means only 10% of the image or so is discarded. 
        self.crop_proportion = crop_proportion
        self.MEAN = np.reshape(np.array([0.485, 0.458, .407]), [1,1,3])
        #TODO: what happens when crop_proportion is none, namely hwen we don't want any augmentation/random cropping?
        #TODO: depending on your implementation of model.predict(): maybe support singleton datasets
        self.train = True
        self.filenames = glob(os.path.join(self.data_dir, glob_pattern))
        self.N = len(self.filenames)
        self.label_dict = {'hartebeest':0, 'deer':1, 'sheep':2}
        self.label_list = ['hartebeest', 'deer', 'sheep']
        self.num_labels=len(self.label_list)
        self.labels = [self.label_dict[f.split('/')[-2]] for f in self.filenames]

    def new_epoch(self):
        #Function finished, resets to a new epoch
        x_y = list(zip(self.filenames, self.labels))
        random.shuffle(x_y)
        self.filenames, self.labels = zip(*x_y)
        self.idx = 0

    def get_next_batch(self):
        #TODO: Implement this!
        if not self.train:
           return np.array([self.load_image(self.filenames[0])])
        return batch, batch_y

    def load_image(self, filename):
        image = misc.imread(filename, 'RGB')
        hw = image.shape[:-1]
        #Think about how you want to resize an image before randomly cropping it to self.h x self.w size
        
         
        resized = resize(crop, ...?)
        return resized

    def random_crop(self, image):
        ###randomly crops the image to correct size 
        s_h = np.random.randint(image.shape[0]-self.h)
        s_w = np.random.randint(image.shape[1]-self.w)
        cropped = image[s_h:s_h+self.h, s_w:s_w+self.w, :]
        return cropped

    def augment_image(self, image):
        image = image - self.MEAN
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        if self.crop_proportion is not None:
            image = self.random_crop(image)
        #Add more augmentation if ya want
        return image

        
