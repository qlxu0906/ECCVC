# -----------------------------------------------------
# Detect Faces Using Ready-made Detector
#
# Author: Liangqi Li
# Creating Date: May 24, 2018
# Latest rectifying: May 24, 2018
# -----------------------------------------------------
import os
import random

import face_recognition
import matplotlib.pyplot as plt


def show_detection_example(im_path):
    """Show a single image and its detecting results"""

    img = face_recognition.load_image_file(im_path)
    face_locations = face_recognition.face_locations(
        img, number_of_times_to_upsample=0, model='cnn')
    print('{} face(s) found in this image.'.format(len(face_locations)))

    fig, ax = plt.subplots()
    ax.imshow(plt.imread(im_path))
    plt.axis('off')
    for face_loc in face_locations:
        # Print the location of each face
        y1, x2, y2, x1 = face_loc
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


def main():

    data_dir = '/home/liliangqi/hdd/datasets/ECCVchallenge/' + \
               'person_search_trainval/train'

    # Randomly pick one image and show the faces detected
    movie_dir = os.path.join(data_dir, random.choice(os.listdir(data_dir)))
    im_dir = os.path.join(movie_dir, 'candidates')
    im_path = os.path.join(im_dir, random.choice(os.listdir(im_dir)))
    show_detection_example(im_path)

if __name__ == '__main__':

    main()
