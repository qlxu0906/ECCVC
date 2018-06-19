# -----------------------------------------------------
# Identify Faces to Match the Cast and the Candidates
#
# Author: Liangqi Li
# Creating Date: Jun 4, 2018
# Latest rectifying: Jun 6, 2018
# -----------------------------------------------------
import os
import random
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import face_recognition

from __init__ import clock_non_return


def crop_image(im_path, box, wh=True):
    """
    Crop image with location indicated by the 'box'
    ---
    param:
        im_path: path to the original image
        box: ndarray or list with shape (4,)
        wh: boolean variable that indicates the coordinates mode of the box
    return:
        ndarray that contain the cropped image
    """

    im = Image.open(im_path)
    if wh:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = box

    cropped_im = im.crop((x1, y1, x2, y2))

    return np.array(cropped_im)


def show_identification_example(root_dir):
    """Compare person samples using face recognition"""

    galleries_df = pd.read_csv(os.path.join(root_dir, 'trainGalleriesDF.csv'))

    data_dir = os.path.join(root_dir, 'train')
    movie = random.choice(os.listdir(data_dir))
    movie_dir = os.path.join(data_dir, movie)

    cast_dir = os.path.join(movie_dir, 'cast')
    cast_imname = random.choice(os.listdir(cast_dir))
    cast_impath = os.path.join(cast_dir, cast_imname)
    cast_img = np.array(Image.open(cast_impath))
    pid = cast_imname[:-4]

    cand_same_pid_df = galleries_df.query('movie==@movie and pid==@pid')
    chosen_index = random.choice(list(cand_same_pid_df.index))
    chosen_se = cand_same_pid_df.loc[chosen_index]
    cand_imname = chosen_se['imname']
    cand_impath = os.path.join(movie_dir, 'candidates', cand_imname)

    cand_person_box = chosen_se['x1': 'del_y'].as_matrix()
    cand_person_img = crop_image(cand_impath, cand_person_box)

    # Show the result
    plt.subplot(121)
    plt.axis('off')
    plt.title(pid)
    plt.imshow(cast_img)
    plt.subplot(122)
    plt.axis('off')
    plt.title(chosen_se['pid'])
    plt.imshow(cand_person_img)
    plt.show()

    cast_face_encoding = face_recognition.face_encodings(cast_img)[0]
    cand_face_encoding = face_recognition.face_encodings(cand_person_img)[0]

    results = face_recognition.compare_faces(
        [cast_face_encoding], cand_face_encoding)

    print(results)


def show_distance_example(root_dir):
    """Output distance between one query and multiple galleries"""

    galleries_df = pd.read_csv(os.path.join(root_dir, 'trainGalleriesDF.csv'))

    data_dir = os.path.join(root_dir, 'train')
    movie = random.choice(os.listdir(data_dir))
    movie_dir = os.path.join(data_dir, movie)

    cast_dir = os.path.join(movie_dir, 'cast')
    cast_imname = random.choice(os.listdir(cast_dir))
    cast_impath = os.path.join(cast_dir, cast_imname)
    cast_img = np.array(Image.open(cast_impath))
    cast_face_encoding = face_recognition.face_encodings(cast_img)[0]

    mv_gallery_df = galleries_df.query('movie==@movie')
    face_distances = np.array([0])

    # Go through all the candidates in this movie
    for index in mv_gallery_df.index:
        cur_imname = mv_gallery_df.loc[index, 'imname']
        cur_impath = os.path.join(movie_dir, 'candidates', cur_imname)
        cur_perosn_box = mv_gallery_df.loc[index, 'x1': 'del_y'].as_matrix()
        cur_person_img = crop_image(cur_impath, cur_perosn_box)
        cur_face_encodings = face_recognition.face_encodings(cur_person_img)

        # Maybe there is no face in a candidate image box
        if len(cur_face_encodings):
            cur_face_encoding = cur_face_encodings[0]
            face_distance = face_recognition.face_distance(
                [cast_face_encoding], cur_face_encoding)
        else:
            face_distance = np.array([1])

        face_distances = np.hstack((face_distances, face_distance))

    face_distances = face_distances[1:]
    similar_indices = np.argsort(face_distances)
    # Pick out the most similar one to the query
    sim_se = mv_gallery_df.iloc[similar_indices[0]]
    sim_impath = os.path.join(movie_dir, 'candidates', sim_se['imname'])
    sim_box = sim_se['x1': 'del_y'].as_matrix()
    sim_img = crop_image(sim_impath, sim_box)

    # Show the query and the picked candidate
    plt.subplot(121)
    plt.axis('off')
    plt.title(cast_imname[:-4])
    plt.imshow(cast_img)
    plt.subplot(122)
    plt.axis('off')
    plt.title(sim_se['pid'])
    plt.imshow(sim_img)
    plt.show()


def match_val_candidates(root_dir):
    """
    Match each cast with a corresponding candidate
    ---
    return:
        matching_result: a dict that contains all casts and their corresponding
                         candidate IDs
    """

    galleries_df = pd.read_csv(os.path.join(root_dir, 'valGalleriesDF.csv'))
    data_dir = os.path.join(root_dir, 'val')

    matching_result = {}
    movies = os.listdir(data_dir)
    for i, movie in enumerate(movies, 1):
        matching_result[movie] = {}
        movie_dir = os.path.join(data_dir, movie)

        cast_dir = os.path.join(movie_dir, 'cast')
        casts = os.listdir(cast_dir)
        for j, cast_imname in enumerate(casts, 1):
            print('Processing movie {}/{}, cast {}/{}'.format(
                i, len(movies), j, len(casts)))
            cast_impath = os.path.join(cast_dir, cast_imname)
            cast_img = np.array(Image.open(cast_impath))
            face_locs = face_recognition.face_locations(cast_img, model='cnn')
            cast_face_encodings = face_recognition.face_encodings(
                cast_img, face_locs)
            if len(cast_face_encodings) == 0:
                # Maybe the detector dose not detect any faces in cast image
                matching_result[movie][cast_imname] = -1
                continue
            cast_face_encoding = cast_face_encodings[0]

            mv_gallery_df = galleries_df.query('movie==@movie')
            face_distances = np.array([0])

            # Go through all the candidates in this movie
            for index in mv_gallery_df.index:
                cur_imname = mv_gallery_df.loc[index, 'imname']
                cur_impath = os.path.join(movie_dir, 'candidates', cur_imname)
                cur_perosn_box = mv_gallery_df.loc[
                                index, 'x1': 'del_y'].as_matrix()
                cur_person_img = crop_image(cur_impath, cur_perosn_box)
                cur_locs = face_recognition.face_locations(cur_person_img,
                                                           model='cnn')
                cur_face_encodings = face_recognition.face_encodings(
                    cur_person_img, cur_locs)

                # Maybe there is no face in a candidate image box
                if len(cur_face_encodings):
                    cur_face_encoding = cur_face_encodings[0]
                    face_distance = face_recognition.face_distance(
                        [cast_face_encoding], cur_face_encoding)
                else:
                    face_distance = np.array([1])

                face_distances = np.hstack((face_distances, face_distance))

            face_distances = face_distances[1:]
            similar_indices = np.argsort(face_distances)
            # Pick out the most similar one to the query
            sim_se = mv_gallery_df.iloc[similar_indices[0]]
            sim_id = sim_se['id']
            matching_result[movie][cast_imname] = int(sim_id)

    file_name = 'matching_val_results.json'
    with open(os.path.join(root_dir, file_name), 'w') as f:
        json.dump(matching_result, f, indent=4)

    return matching_result


def fix_matching_result(root_dir):
    """Fill the missing value of the matching results"""

    file_name = 'matching_val_results.json'
    with open(os.path.join(root_dir, file_name), 'r') as f:
        matching_result = json.load(f)

    missing_movies = ['tt0112573', 'tt0096446', 'tt0110604']
    missing_casts = ['nm0732703.jpg', 'nm0730053.jpg', 'nm0724084.jpg']

    for movie, cast in zip(missing_movies, missing_casts):
        if movie in matching_result.keys():
            if cast not in matching_result[movie].keys():
                matching_result[movie][cast] = -1

    with open(os.path.join(root_dir, file_name), 'w') as f:
        json.dump(matching_result, f, indent=4)


def evaluate_matching_result(root_dir, num_result):
    """Evaluate the accuracy of the matching result"""

    galleries_df = pd.read_csv(os.path.join(root_dir, 'valGalleriesDF.csv'))
    file_name = 'matching_val_results.json'
    with open(os.path.join(root_dir, file_name), 'r') as f:
        matching_result = json.load(f)

    correct = 0
    total = 0

    for i in range(num_result):
        for movie in matching_result.keys():
            for cast in matching_result[movie]:
                total += 1
                cast_pid = cast[:-4]
                cand_ids = matching_result[movie][cast]
                if cand_ids == -1:
                    continue
                cand_id = cand_ids[i]
                assert cand_id != -1
                cand_df = galleries_df.query('movie==@movie and id==@cand_id')
                assert cand_df.shape[0] == 1
                cand_pid = cand_df.iloc[0]['pid']
                if cast_pid == cand_pid:
                    correct += 1

        accuracy = correct / total * 100
        print('{}-th matching accuracy on val dataset is {:.2f}%.'.format(
            i + 1, accuracy))


@clock_non_return
def main():

    data_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' +\
               'person_search_trainval'
    # show_identification_example(data_dir)
    # show_distance_example(data_dir)
    # match_val_candidates(data_dir)
    # fix_matching_result(data_dir)
    evaluate_matching_result(data_dir, 5)


if __name__ == '__main__':

    main()
