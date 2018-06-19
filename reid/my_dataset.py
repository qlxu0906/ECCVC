
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def find_classes(root):
    classes = [d for d in root.pid]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class myDataset(Dataset):
    def __init__(self, df,path,name,train=True, transform=None) :
        
        self.train = train
        #df = pd.read_csv(csv_file)
        self.root = df
        classes, class_to_idx = find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        #self.root_dir = root_dir
        self.transform = transform
        self.path = path
        self.name = name
    def __getitem__(self, index):
        movie_name,img_name,id,x1,y1,del_x,del_y,label = self.root.iloc[index]
        label_idx=self.class_to_idx[label]
        img_path = self.path + '/'+self.name+'/'+ movie_name + '/candidates/'+img_name
        bbox = (x1,y1,x1+del_x,y1+del_y)
        img = Image.open(img_path)
        image = img.crop(bbox)
        #landmarks = self.root.as_matrix()[index, 1:].astype('float')
        #landmarks = np.reshape(landmarks,newshape=(-1,2))
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

    def __len__(self):
        return len(self.root)

'''
trans = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

eccv_dataset = myDataset(csv_file='/home/xuqiling/eccvc/dataset/trainGalleriesDF.csv', transform= trans)
loader = DataLoader(dataset = eccv_dataset, batch_size=32,shuffle=True,num_workers=16)
'''