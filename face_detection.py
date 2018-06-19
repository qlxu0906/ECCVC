# -----------------------------------------------------
# Detect Faces Using Ready-made Detector
#
# Author: Liangqi Li
# Creating Date: May 24, 2018
# Latest rectifying: Jun 4, 2018
# -----------------------------------------------------
import os
import random

import face_recognition
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import torch

from __init__ import clock_non_return
import SFD.detection as sfd_detection
import SFD.net_s3fd as net_s3fd


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--tool', default='face_rec', type=str)
    parser.add_argument('--model_dir', default='', type=str)

    args = parser.parse_args()

    return args


def show_faces(im_path, indices, f_boxes):
    """
    Show faces with given boxes
    ---
    param:
        im_path: the path to th image that need to be shown
        indices: the indices of `f_boxes` in the DataFrame
        f_boxes: ndarray with shape (N, 4) that represents the locations of the
                 faces, the order of coordinates is (x1, y1, w, h)
    """

    fig, ax = plt.subplots()
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    for index, f_box in zip(indices, f_boxes):
        x1, y1, w, h = f_box
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False,
                                   edgecolor='#4CAF50', linewidth=3.5))
        ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False,
                                   edgecolor='white', linewidth=1))
        ax.text(x1 + 5, y1 - 15, index,
                bbox=dict(facecolor='#4CAF50', linewidth=0),
                fontsize=20, color='white')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def show_detection_example(root_dir, tool='face_rec', model_path=None):
    """Show a single image and its detecting results"""

    # Randomly pick one image and show the faces detected
    movie_dir = os.path.join(root_dir, random.choice(os.listdir(root_dir)))
    im_dir = os.path.join(movie_dir, 'cast')
    im_path = os.path.join(im_dir, random.choice(os.listdir(im_dir)))

    if tool == 'face_rec':
        img = face_recognition.load_image_file(im_path)
        face_locations = face_recognition.face_locations(
            img, number_of_times_to_upsample=0, model='cnn')
    elif tool == 'sfd':
        assert model_path is not None
        net = net_s3fd.s3fd()
        net.load_state_dict(torch.load(model_path))
        net.cuda()
        net.eval()
        face_locations = sfd_detection.output(net, im_path)
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


def show_person_face(im_path, f_box, p_box):
    """
    Show the person and the face in the image
    ---
    param:
        f_box: a ndarray that represents the location of the face
        p_box: a ndarray that represents the location of the person
    """

    fig, ax = plt.subplots()
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    ax.add_patch(plt.Rectangle(
        (f_box[0], f_box[1]), f_box[2] - f_box[0], f_box[3] - f_box[1],
        fill=False, edgecolor='#4CAF50', linewidth=3.5))
    ax.add_patch(plt.Rectangle(
        (f_box[0], f_box[1]), f_box[2] - f_box[0], f_box[3] - f_box[1],
        fill=False, edgecolor='white', linewidth=1))
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2] - p_box[0], p_box[3] - p_box[1],
        fill=False, edgecolor='#66D9EF', linewidth=3.5))
    ax.add_patch(plt.Rectangle(
        (p_box[0], p_box[1]), p_box[2] - p_box[0], p_box[3] - p_box[1],
        fill=False, edgecolor='white', linewidth=1))
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


def assign_id_to_face(im_path, g_df, tool, net=None):
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
        assert net is not None
        face_locations = sfd_detection.output(net, im_path)
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
            # Show current person and face
            show_person_face(im_path, face_box, person_box)

            if face_in_person(face_box, person_box):
                # Return (x1, y1, w, h)
                face_box[2] -= face_box[0]
                face_box[3] -= face_box[1]
                faces.append((face_box, pid))
                # TODO: One face contained by more than one person
                break

    return faces


def get_cand_face_annotations(anno_dir, tool='face_rec', model_path=None):
    """Get candidate face annotations as DataFrame"""

    galleries = pd.read_csv(os.path.join(anno_dir, 'trainGalleriesDF.csv'))
    data_dir = os.path.join(anno_dir, 'train')
    movies = os.listdir(data_dir)

    if tool == 'sfd':
        assert model_path is not None
        net = net_s3fd.s3fd()
        net.load_state_dict(torch.load(model_path))
        net.cuda()
        net.eval()
    else:
        net = None

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
            g_faces = assign_id_to_face(g_impath, g_df, tool, net)
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


def get_cast_face_annotations(anno_dir, tool='face_rec', model_path=None):
    """Get candidate face annotations as DataFrame"""

    data_dir = os.path.join(anno_dir, 'train')
    movies = os.listdir(data_dir)

    if tool == 'sfd':
        assert model_path is not None
        net = net_s3fd.s3fd()
        net.load_state_dict(torch.load(model_path))
        net.cuda()
        net.eval()
    else:
        net = None

    f_movies = []
    f_imnames = []
    f_boxes = np.zeros((1, 4), dtype=np.int32)
    f_pids = []

    for i, movie in enumerate(movies, 1):
        candidates_dir = os.path.join(data_dir, movie, 'cast')
        q_imnames = os.listdir(candidates_dir)
        for j, q_imname in enumerate(q_imnames, 1):
            print('Movie {}/{}, image {}/{}'.format(
                i, len(movies), j, len(q_imnames)))
            q_impath = os.path.join(candidates_dir, q_imname)
            if tool == 'face_rec':
                img = face_recognition.load_image_file(q_impath)
                q_faces = face_recognition.face_locations(
                    img, number_of_times_to_upsample=0, model='cnn')
            elif tool == 'sfd':
                assert net is not None
                q_faces = sfd_detection.output(net, q_impath)
            else:
                raise KeyError(tool)

            for face in q_faces:
                if tool == 'face_rec':
                    y1, x2, y2, x1 = face
                elif tool == 'sfd':
                    x1, y1, x2, y2 = face
                else:
                    raise KeyError(tool)
                f_box = np.array([x1, y1, x2, y2])
                f_box[2] -= f_box[0]
                f_box[3] -= f_box[1]
                f_boxes = np.vstack((f_boxes, f_box))
                f_id = q_imname[:-4]
                f_pids.append(f_id)
                f_movies.append(movie)
                f_imnames.append(q_imname)

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
        anno_dir, 'trainCastFacesDF_{}.csv'.format(tool)), index=False)


def remove_outliers(anno_dir):
    """Remove outlier samples that appear twice in an image"""

    # Process faces produced by `face_rec`
    faces_rec = pd.read_csv(os.path.join(
        anno_dir, 'trainCastFacesDF_face_rec.csv'))
    outliers_rec = []

    for movie in set(faces_rec['movie']):
        mv_df = faces_rec.query('movie==@movie')
        for im_name in set(mv_df['imname']):
            im_df = mv_df.query('imname==@im_name')
            if im_df.shape[0] > 1:
                im_path = os.path.join(
                    anno_dir, 'train', movie, 'cast', im_name)
                show_faces(im_path, list(im_df.index),
                           im_df.loc[:, 'x1': 'del_y'].as_matrix())
                for _ in range(im_df.shape[0] - 1):
                    outliers_rec.append(int(input('Enter the outlier: ')))

    for outlier in outliers_rec:
        faces_rec.drop(outlier, inplace=True)
    faces_rec.index = range(faces_rec.shape[0])
    faces_rec.to_csv(os.path.join(
        anno_dir, 'trainCastFacesDF_face_rec.csv'), index=False)


@clock_non_return
def main():

    opt = parse_args()
    show_detection_example(os.path.join(opt.data_dir, 'train'),
                           opt.tool, opt.model_dir)
    get_cand_face_annotations(opt.data_dir, opt.tool, opt.model_dir)
    get_cast_face_annotations(opt.data_dir)
    remove_outliers(opt.data_dir)


if __name__ == '__main__':

    main()
