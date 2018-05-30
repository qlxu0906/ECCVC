# -----------------------------------------------------
# Transform Annotation Information to Structured Data
#
# Author: Liangqi Li
# Creating Date: May 30, 2018
# Latest rectifying: May 30, 2018
# -----------------------------------------------------
import os
import json

import numpy as np
import pandas as pd


def process_train_set(anno_dir):
    """Read training annotation files and transfer into DataFrame"""

    q_movies = []
    q_ids = []
    q_ims = []

    g_movies = []
    g_ids = []
    g_ims = []
    g_boxes = np.zeros((1, 4), dtype=np.int32)
    g_pids = []

    with open(os.path.join(anno_dir, 'train.json'), 'r') as f:
        train = json.load(f)

    for movie, images in train.items():
        casts = images['cast']
        candidates = images['candidates']
        for cast in casts:
            id_num = cast['id'].split('_')[-1]
            assert id_num == cast['label']
            im_name = cast['img'].split('/')[-1]
            q_movies.append(movie)
            q_ids.append(id_num)
            q_ims.append(im_name)
        for cand in candidates:
            id_num = cand['id'].split('_')[-1]
            im_name = cand['img'].split('/')[-1]
            box = np.array(cand['bbox'])
            pid = cand['label']
            g_movies.append(movie)
            g_ids.append(id_num)
            g_ims.append(im_name)
            g_boxes = np.vstack((g_boxes, box))
            g_pids.append(pid)

    queries = pd.DataFrame({'movie': q_movies, 'imname': q_ims, 'pid': q_ids})

    # Remove the first row
    g_boxes = g_boxes[1:]
    galleries = pd.DataFrame(g_boxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    galleries['movie'] = g_movies
    galleries['id'] = g_ids
    galleries['imname'] = g_ims
    galleries['pid'] = g_pids

    # Indicate the order of the column names
    ordered_columns = ['movie', 'imname', 'id', 'x1', 'y1', 'del_x', 'del_y',
                       'pid']
    galleries = galleries[ordered_columns]

    # Save the DataFrames to csv files
    queries.to_csv(os.path.join(anno_dir, 'trainQueriesDF.csv'), index=False)
    galleries.to_csv(os.path.join(anno_dir, 'trainGalleriesDF.csv'),
                     index=False)


def process_val_set(anno_dir):
    """Read validate annotation files and transfer into DataFrame"""

    q_movies = []
    q_ids = []
    q_ims = []

    g_movies = []
    g_ids = []
    g_ims = []
    g_boxes = np.zeros((1, 4), dtype=np.int32)
    g_pids = []

    with open(os.path.join(anno_dir, 'val.json'), 'r') as f:
        val = json.load(f)

    for movie, images in val.items():
        casts = images['cast']
        candidates = images['candidates']
        for cast in casts:
            id_num = cast['id'].split('_')[-1]
            assert id_num == cast['label']
            im_name = cast['img'].split('/')[-1]
            q_movies.append(movie)
            q_ids.append(id_num)
            q_ims.append(im_name)
        for cand in candidates:
            id_num = cand['id'].split('_')[-1]
            im_name = cand['img'].split('/')[-1]
            box = np.array(cand['bbox'])
            pid = cand['label']
            g_movies.append(movie)
            g_ids.append(id_num)
            g_ims.append(im_name)
            g_boxes = np.vstack((g_boxes, box))
            g_pids.append(pid)

    queries = pd.DataFrame({'movie': q_movies, 'imname': q_ims, 'pid': q_ids})

    # Remove the first row
    g_boxes = g_boxes[1:]
    galleries = pd.DataFrame(g_boxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    galleries['movie'] = g_movies
    galleries['id'] = g_ids
    galleries['imname'] = g_ims
    galleries['pid'] = g_pids

    # Indicate the order of the column names
    ordered_columns = ['movie', 'imname', 'id', 'x1', 'y1', 'del_x', 'del_y',
                       'pid']
    galleries = galleries[ordered_columns]

    # Save the DataFrames to csv files
    queries.to_csv(os.path.join(anno_dir, 'valQueriesDF.csv'), index=False)
    galleries.to_csv(os.path.join(anno_dir, 'valGalleriesDF.csv'),
                     index=False)


def main():
    anno_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' +\
               'person_search_trainval'
    process_train_set(anno_dir)
    process_val_set(anno_dir)


if __name__ == '__main__':

    main()
