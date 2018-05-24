# -----------------------------------------------------
# Analyse Annotations of the Dataset
#
# Author: Liangqi Li
# Creating Date: May 19, 2018
# Latest rectifying: May 20, 2018
# -----------------------------------------------------
import os
import json


def create_example_submission(root_dir):
    """Create an example of submission to the challenge"""

    val_gt_file = os.path.join(root_dir, 'val_label.json')

    with open(val_gt_file, 'r') as f1:
        val_gt = json.load(f1)

    f2 = open(os.path.join(root_dir, 'trans_gt.txt'), 'a')
    for key, value in val_gt.items():
        f2.write(key)
        f2.write(' ')
        value_length = len(value)
        for i, item in enumerate(value):
            f2.write(item)
            # Do NOT write a comma for the last one
            if i != value_length - 1:
                f2.write(',')
        f2.write('\n')
    f2.close()


def main():
    root_dir = '/Users/liliangqi/Desktop/ECCVchallenge/' + \
               'person_search_eval_example/evaluation/'
    create_example_submission(root_dir)


if __name__ == '__main__':

    main()
