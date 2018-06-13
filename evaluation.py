# -----------------------------------------------------
# Evaluate the Final Search Result
#
# Author: Liangqi Li
# Creating Date: Jun 8, 2018
# Latest rectifying: Jun 13, 2018
# -----------------------------------------------------
import os
import random

from __init__ import clock_non_return

import matplotlib.pyplot as plt
import pandas as pd


def show_hard_positive(im_path, p_box, order, total, save_dir):
    """
    Show the positive person in the image that may be hard to identify
    ---
    param:
        p_box: a ndarray that represents the location of the person
    """

    fig, ax = plt.subplots()
    plt.title('Order: {}/{}'.format(order, total))
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2], p_box[3], fill=False,
        edgecolor='#66D9EF', linewidth=3.5))
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2], p_box[3], fill=False,
        edgecolor='white', linewidth=1))
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'hard_{}.jpg'.format(order)),
                bbox_inches='tight')
    plt.close()


def save_query(im_path, p_box, save_dir, index, matched):
    """
    Save the sample used as query in the image
    ---
    param:
        p_box: a ndarray that represents the location of the person
        matched: boolean variable to judge if this query is positive or not
    """

    fig, ax = plt.subplots()
    plt.title('Query #{}, matched: {}'.format(index, matched))
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2], p_box[3], fill=False,
        edgecolor='#66D9EF', linewidth=3.5))
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2], p_box[3], fill=False,
        edgecolor='white', linewidth=1))
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'query_{}.jpg'.format(index)),
                bbox_inches='tight')
    plt.close()


def evaluate_final_result():
    root_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' + \
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
        assert cur_id in val_gallery_df['id'].values
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


@clock_non_return
def show_unmatched_ones(shown_num):
    """Show the positive samples sorted in the rear"""

    root_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' + \
               'person_search_trainval'
    data_dir = os.path.join(root_dir, 'val')
    result_file = os.path.join(root_dir, 'result.txt')
    val_gallery_df = pd.read_csv(os.path.join(root_dir, 'valGalleriesDF.csv'))
    save_dir = os.path.join(root_dir, 'observation')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(result_file, 'r') as f:
        lines = f.readlines()

    # line = random.choice(lines)
    for line in lines:
        cast, cands = line.split(' ')
        movie, pid = cast.split('_')
        cands = cands.split(',')

        movie_save_dir = os.path.join(save_dir, movie)
        if not os.path.exists(movie_save_dir):
            os.mkdir(movie_save_dir)
        cast_save_dir = os.path.join(movie_save_dir, pid)
        if not os.path.exists(cast_save_dir):
            os.mkdir(cast_save_dir)

        cast_imname = pid + '.jpg'
        cast_impath = os.path.join(data_dir, movie, 'cast', cast_imname)
        plt.imshow(plt.imread(cast_impath))
        # plt.imread(cast_impath)
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(cast_save_dir, cast_imname),
                    bbox_inches='tight')
        plt.close()

        for i in range(5):
            query = cands[i]
            cur_id = int(query.split('_')[-1])
            assert cur_id in val_gallery_df['id'].values
            cur_df = val_gallery_df.query('movie==@movie and id==@cur_id')
            cur_imname = cur_df.iloc[0]['imname']
            cur_pid = cur_df.iloc[0]['pid']
            cur_pbox = cur_df.loc[:, 'x1': 'del_y'].as_matrix()[0]
            cur_impath = os.path.join(
                data_dir, movie, 'candidates', cur_imname)
            cur_matched = cur_pid == pid
            save_query(cur_impath, cur_pbox, cast_save_dir, i+1, cur_matched)

        hard_index = 0
        for i, cand in enumerate(cands[::-1]):
            if hard_index > shown_num:
                break
            cur_id = int(cand.split('_')[-1])
            assert cur_id in val_gallery_df['id'].values
            cur_df = val_gallery_df.query('movie==@movie and id==@cur_id')
            cur_imname = cur_df.iloc[0]['imname']
            cur_pid = cur_df.iloc[0]['pid']
            cur_pbox = cur_df.loc[:, 'x1': 'del_y'].as_matrix()[0]
            cur_impath = os.path.join(
                data_dir, movie, 'candidates', cur_imname)

            # If we find the positive, then show or save it
            if pid == cur_pid:
                hard_index += 1
                show_hard_positive(cur_impath, cur_pbox, len(cands) - i,
                                   len(cands), cast_save_dir)


if __name__ == '__main__':

    # evaluate_final_result()
    # crop_result(730)
    show_unmatched_ones(5)
