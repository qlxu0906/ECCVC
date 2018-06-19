from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense
from my_dataset import myDataset
import pandas as pd
import json
import random
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/xql/eccv/dataset/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------

#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir

data_path = '/home/xuqiling/eccvc/dataset'
val_data = pd.read_csv('/home/xuqiling/eccvc/dataset/valGalleriesDF.csv',dtype={'id':str})

val_no_others = val_data[val_data.pid!='others']
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('../model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            #print(f.size())
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features
###########################movie###############
movies = [d for d in val_data.movie]
movies = list(set(movies))
movies.sort()

#################################################

use_gpu = torch.cuda.is_available()

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(739)
else:
    model_structure = ft_net(739)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
gallery_feature = torch.FloatTensor()
query_feature = torch.FloatTensor()
gallery_label = []
query_label = []
gallery_id = []
query_id = []
query_pick_label = []
movies = [d for d in val_data.movie]
movies = list(set(movies))
movies.sort()


for i in range(len(movies)):
    movie = movies[i]
    val_data_gallery = val_data[(val_data.movie == movie)]
    image_datasets = {}
    image_datasets['gallery'] = myDataset(val_data_gallery, path=data_path,
                                          name='val', transform=data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=4) for x in ['gallery']}
    gallery_label.extend(val_data_gallery.iloc[:, 7].values)
    gallery_id.extend((val_data_gallery['movie'] + '_' + val_data_gallery['id']).values)
    gallery_feature =  torch.cat((gallery_feature,(extract_feature(model,dataloaders['gallery']))),0)
    result={'gallery_f':gallery_feature.numpy(),'gallery_id':gallery_id,'gallery_label':gallery_label}
scipy.io.savemat('pytorch_result.mat',result)

