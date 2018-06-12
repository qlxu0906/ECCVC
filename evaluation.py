# -----------------------------------------------------
# Evaluate the Final Search Result
#
# Author: Liangqi Li
# Creating Date: Jun 8, 2018
# Latest rectifying: Jun 8, 2018
# -----------------------------------------------------
import os
import random

import pandas as pd


def evaluate_final_result():

    root_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' +\
               'person_search_trainval'
    result_file = os.path.join(root_dir, 'result.txt')
    val_gallery_df = pd.read_csv(os.path.join(root_dir, 'valGalleriesDF.csv'))

    with open(result_file, 'r') as f:
        lines = f.readlines()

    line = random.choice(lines)
    cast, cands = line.split(' ')
    movie, pid = cast.split('_')
    for i, cand in enumerate(cands.split(','), 1):
        cur_id = int(cand.split('_')[-1])
        cur_df = val_gallery_df.query('movie==@movie and id==@cur_id')
        cur_pid = cur_df.iloc[0]['pid']
        cur_match = pid == cur_pid
        print('Candidate {}, matching: {}'.format(i, cur_match))

    print('Debug')


def crop_result(length):
    root_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' + \
               'person_search_trainval'
    result_file = os.path.join(root_dir, 'result.txt')
    new_file = os.path.join(root_dir, 'new_result.txt')

    with open(result_file, 'r') as f1:
        lines = f1.readlines()

    f2 = open(new_file, 'a')
    for line in lines:
        cast, cands = line.split(' ')
        f2.write(cast)
        f2.write(' ')
        for i, cand in enumerate(cands.split(','), 1):
            if i < length:
                f2.write(cand)
                f2.write(',')
            elif i == length:
                f2.write(cand)
                f2.write('\n')
            else:
                break
    f2.close()


if __name__ == '__main__':

    evaluate_final_result()
    crop_result(730)
