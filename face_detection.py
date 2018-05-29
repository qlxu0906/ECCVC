# -----------------------------------------------------
# Detect Faces Using Ready-made Detector
#
# Author: Liangqi Li
# Creating Date: May 24, 2018
# Latest rectifying: May 29, 2018
# -----------------------------------------------------
import argparse
import os
import random

import face_recognition
import matplotlib.pyplot as plt
import sfd_demo

from __init__ import clock_non_return


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


@clock_non_return
def main():
    opt = parse_args()
    show_detection_example(opt.data_dir, opt.tool)


if __name__ == '__main__':
    main()
