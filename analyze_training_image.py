import glob
import os
import argparse
from helpers.analysis.anchors import kmeans
import numpy as np
import random
from os.path import join


def main(augmented_data_path, output_path):

    # Convert to absolute paths
    augmented_data_path = os.path.abspath(augmented_data_path)
    output_path = os.path.abspath(output_path)

    # Create folders if not exist
    os.makedirs(output_path, exist_ok=True)

    # Calculate and save anchors
    augmented_images_paths = glob.iglob(os.path.join(augmented_data_path, "*.png"))
    annotation_dims = []

    for augmented_image_path in augmented_images_paths:

        augmented_image_path = augmented_image_path.replace('.jpg', '.txt')
        augmented_image_path = augmented_image_path.replace('.png', '.txt')

        if not os.path.isfile(augmented_image_path):
            continue
        f2 = open(augmented_image_path)
        for line in f2.readlines():
            line = line.rstrip('\n')
            w, h = line.split(' ')[3:]
            annotation_dims.append(tuple(map(float, (w, h))))
    annotation_dims = np.array(annotation_dims)

    eps = 0.005
    num_clusters = 1 # NOTE: Doors are rarely overlapping

    if num_clusters == 0:
        for num_clusters in range(1, 11):  # we make 1 through 10 clusters
            anchor_file = join(output_path, 'anchors%d.txt' % num_clusters)

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, eps, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join(output_path, 'anchors%d.txt' % num_clusters)
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file)
        print('centroids.shape', centroids.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the training data for YOLO training.')
    parser.add_argument('-i', '--input_path', default='training_data', type=str, help='Path to your raw image data and bounding boxes')
    parser.add_argument('-o', '--output_path', default='door', help='Path where augmented images will be saved')
    args = parser.parse_args()

    main(args.input_path, args.output_path)
