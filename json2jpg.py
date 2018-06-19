# -----------------------------------------------------
# Analyse Annotations of the Dataset
#
# Author: Qiling Xu
# Creating Date: May 25, 2018
# Latest rectifying: May 25, 2018
# -----------------------------------------------------

import json
import os
import os.path

from PIL import Image


def crop(images, name):
    for i in images.keys():
        print(i)
        for j in range(len(images[i]["candidates"])):
            fn = images[i]["candidates"][j]["img"]

            cropmake(fn, images[i]['candidates'][j]['bbox'],
                     images[i]['candidates'][j]['label'], name)


def cropmake(fn, bbox, label, name):
    loc = 'E:/study/ECCVchallenge/person_search_trainval/'
    newfn = fn.replace('.jpg', '_' + label + '.jpg')
    im = Image.open(loc + '/' + name + '/' + fn)
    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]
    bbox = (left, top, left + width, top + height)
    bbox = tuple(bbox)
    try:
        newim = im.crop(bbox)
        newloc = loc + 'newImage/' + name + '/'
        crop_fn = newloc + newfn
        File_Path = os.path.dirname(crop_fn)
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        newim.save(crop_fn)

    except SystemError:
        print("Error: error finding or failure cutting")


def main():

    path = 'E:/study/ECCVchallenge/person_search_trainval/'
    fullpath_t = path + 'train.json'
    fp_t = open(fullpath_t, 'r')
    images_t = json.load(fp_t)
    fp_t.close()
    crop(images_t, 'train')

    fullpath_v = path + 'val.json'
    fp_v = open(fullpath_v, 'r')
    images_v = json.load(fp_v)
    fp_v.close()
    crop(images_v, 'val')


if __name__ == '__main__':

    main()