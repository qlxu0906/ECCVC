# -----------------------------------------------------
# Detect Faces Using Ready-made Detector
#
# Author: Liangqi Li
# Creating Date: May 24, 2018
# Latest rectifying: May 29, 2018
# -----------------------------------------------------
import os
import random

import face_recognition
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

from __init__ import clock_non_return
import sfd_demo


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--tool', default='face_rec', type=str)

    args = parser.parse_args()

    return args


def show_detection_example(root_dir, tool='face_rec'):
    """Show a single image and its detecting results"""

    # Randomly pick one image and show the faces detected
    movie_dir = os.path.join(root_dir, random.choice(os.listdir(root_dir)))
    im_dir = os.path.join(movie_dir, 'candidates')
    im_path = os.path.join(im_dir, random.choice(os.listdir(im_dir)))

    if tool == 'face_rec':
        img = face_recognition.load_image_file(im_path)
        face_locations = face_recognition.face_locations(
            img, number_of_times_to_upsample=0, model='cnn')
    elif tool == 'sfd':
        face_locations = sfd_demo.demo(im_path)
    else:
        raise KeyError(tool)
    print('{} face(s) found in this image.'.format(len(face_locations)))

    fig, ax = plt.subplots()
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    for face_loc in face_locations:
        # Print the location of each face
        if tool == 'face_rec':
            y1, x2, y2, x1 = face_loc
        elif tool == 'sfd':
            x1, y1, x2, y2 = face_loc
        else:
            raise KeyError(tool)
        print('A face is located at [({}, {}), ({}, {})]'.format(
            x1, y1, x2, y2))

        # Show the results
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor='#4CAF50', linewidth=3.5))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor='white', linewidth=1))
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def face_in_person(f_box, p_box):
    """
    Judge if the `f_box` is contained by the `p_box`
    ---
    param:
        f_box: a ndarray that represents the location of the face
        p_box: a ndarray that represents the location of the person
    """

    return f_box[0] > p_box[0] and f_box[1] > p_box[1] and \
        f_box[2] < p_box[2] and f_box[3] < p_box[3]


def assign_id_to_face(im_path, g_df, tool):
    """
    Detect faces and assign them with IDs from person
    ---
    param:
        im_path: absolute path of the image to be detected
        g_df: DataFrame that only contain information about a single image
        tool: type of detector
    return:
        faces: a list that contains face locations and corresponding ID
    """

    if tool == 'face_rec':
        img = face_recognition.load_image_file(im_path)
        face_locations = face_recognition.face_locations(
            img, number_of_times_to_upsample=0, model='cnn')
    elif tool == 'sfd':
        face_locations = sfd_demo.demo(im_path)
    else:
        raise KeyError(tool)

    faces = []
    person_boxes = g_df.loc[:, 'x1': 'del_y'].as_matrix()
    person_boxes[:, 2] += person_boxes[:, 0]
    person_boxes[:, 3] += person_boxes[:, 1]
    pids = g_df['pid'].values

    for face_loc in face_locations:
        if tool == 'face_rec':
            y1, x2, y2, x1 = face_loc
        elif tool == 'sfd':
            x1, y1, x2, y2 = face_loc
        else:
            raise KeyError(tool)

        face_box = np.array([x1, y1, x2, y2])
        for person_box, pid in zip(person_boxes, pids):
            # # Show current person and face
            # fig, ax = plt.subplots()
            # ax.imshow(plt.imread(im_path))
            # plt.axis('off')
            # ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
            #                            edgecolor='#4CAF50', linewidth=3.5))
            # ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
            #                            edgecolor='white', linewidth=1))
            # ax.add_patch(plt.Rectangle(
            #     (person_box[0], person_box[1]), person_box[2] - person_box[0],
            #     person_box[3] - person_box[1], fill=False, edgecolor='#66D9EF',
            #     linewidth=3.5))
            # ax.add_patch(plt.Rectangle(
            #     (person_box[0], person_box[1]), person_box[2] - person_box[0],
            #     person_box[3] - person_box[1], fill=False, edgecolor='white',
            #     linewidth=1))
            # plt.tight_layout()
            # plt.show()
            # plt.close(fig)

            if face_in_person(face_box, person_box):
                # Return (x1, y1, w, h)
                face_box[2] -= face_box[0]
                face_box[3] -= face_box[1]
                faces.append((face_box, pid))
                # TODO: One face contained by more than one person
                break

    return faces


def get_face_annotations(anno_dir, tool='face_rec'):
    """Get face annotations as DataFrame"""

    galleries = pd.read_csv(os.path.join(anno_dir, 'trainGalleriesDF.csv'))
    data_dir = os.path.join(anno_dir, 'train')
    movies = os.listdir(data_dir)

    images_with_no_person = []
    f_movies = []
    f_imnames = []
    f_boxes = np.zeros((1, 4), dtype=np.int32)
    f_pids = []

    for i, movie in enumerate(movies, 1):
        candidates_dir = os.path.join(data_dir, movie, 'candidates')
        g_imnames = os.listdir(candidates_dir)
        for j, g_imname in enumerate(g_imnames, 1):
            print('Movie {}/{}, image {}/{}'.format(
                i, len(movies), j, len(g_imnames)))
            if g_imname not in galleries['imname'].values:
                images_with_no_person.append(g_imname)
                continue
            g_df = galleries.query('movie==@movie and imname==@g_imname')
            g_impath = os.path.join(candidates_dir, g_imname)
            g_faces = assign_id_to_face(g_impath, g_df, tool)
            if len(g_faces) == 0:
                continue
            for g_face in g_faces:
                f_box, f_id = g_face
                f_boxes = np.vstack((f_boxes, f_box))
                f_pids.append(f_id)
                f_movies.append(movie)
                f_imnames.append(g_imname)

    # Remove the first row
    f_boxes = f_boxes[1:]
    faces = pd.DataFrame(f_boxes, columns=['x1', 'y1', 'del_x', 'del_y'])
    faces['movie'] = f_movies
    faces['imname'] = f_imnames
    faces['pid'] = f_pids

    # Indicate the order of the column names
    ordered_columns = ['movie', 'imname', 'x1', 'y1', 'del_x', 'del_y', 'pid']
    faces = faces[ordered_columns]

    # Save the DataFrames to csv files
    faces.to_csv(os.path.join(
        anno_dir, 'trainFacesDF_{}.csv'.format(tool)), index=False)


@clock_non_return
def main():
    opt = parse_args()
    # show_detection_example(os.path.join(opt.data_dir, 'train'), opt.tool)
    get_face_annotations(opt.data_dir, opt.tool)

if __name__ == '__main__':

    main()
