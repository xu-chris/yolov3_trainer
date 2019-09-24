import glob
import os
import argparse
import shutil
from helpers.augmentation.augmentor import start_pipeline
import numpy as np
import random
from os.path import join

images_width = 608.
images_height = 608.


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= images_width / 32.
        anchors[i][1] *= images_height / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('{},{}, '.format(int(anchors[i, 0]), int(anchors[i, 1])))

    # there should not be comma after last anchor, that's why
    f.write('{},{}'.format(int(anchors[sorted_indices[-1:], 0]), int(anchors[sorted_indices[-1:], 1])))

    # f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def divide_data_set(image_data_path, percentage_validation=0.1, percentage_test=0.1):

    np.random.seed(10101)
    np.random.shuffle(image_data_path)
    np.random.seed(None)
    num_val = int(len(image_data_path) * percentage_validation)
    num_test = int(len(image_data_path) * percentage_test)
    num_train = len(image_data_path) - num_val - num_test

    print('Dataset - total: {}, train: {}, validation: {}, test: {}'.format(len(image_data_path), num_train, num_val, num_test))

    return image_data_path[:num_train], image_data_path[num_train:num_train + num_val], image_data_path[num_train + num_val:]


# Load file and return text from file
def load_file(path):
    file = open(path, "r")

    lines = None
    if file.mode == 'r':
        lines = file.readlines()
        for line, index in zip(lines, range(len(lines))):
            lines[index] = line.rstrip("\n")

    print("File {} contains {} bounding boxes.".format(path, len(lines)))

    file.close()

    return lines


def delete_bottleneck():
    if os.path.isfile("bottlenecks.npz"):
        os.remove("bottlenecks.npz")
        print('Bottlenecks file deleted')
        return
    print('No bottlenecks file found')


# Get text file name for image file
def get_formatted_box_string_for(filepath):
    # Load data from bounding box text file and preformat it
    path, file_extension = os.path.splitext(filepath)
    bounding_box_file = path + ".txt"

    # Skip data without bounding boxes
    if not os.path.isfile(bounding_box_file):
        print('File {} does not have any bounding box file. Skipping...'.format(filepath))
        return

    bounding_boxes = load_file(bounding_box_file)

    if bounding_boxes is None or bounding_boxes == "" or bounding_boxes == []:
        print('File {} does not have any bounding boxes. File is empty. Skipping...'.format(filepath))
        return

    bounding_boxes_string = ""
    for bounding_box in bounding_boxes:
        # separate values with comma, remove spaces
        class_id, x_center, y_center, width, height = bounding_box.split(" ")
        x_min = int(min(max(0, round(float(x_center) * images_width - (float(width) * images_width / 2))), images_width))
        y_min = int(min(max(0, round(float(y_center) * images_height - (float(height) * images_height / 2))), images_height))
        x_max = int(min(max(0, round(float(x_center) * images_width + (float(width) * images_width / 2))), images_width))
        y_max = int(min(max(0, round(float(y_center) * images_height + (float(height) * images_height / 2))), images_height))
        bounding_boxes_string += " {},{},{},{},{}".format(x_min, y_min, x_max, y_max, class_id)

    # Combine image file name with information
    return os.path.abspath(filepath) + bounding_boxes_string


# Save converted annotation file
def save_file(path, lines):
    file = open(path, "w+")

    for line in lines:

        # Skip empty lines
        if line == None:
            continue
        file.write(line + '\n')

    file.close()


def main(image_data_path, augmented_data_path, output_path, remove_bottleneck = False):

    # Convert to absolute paths
    image_data_path = os.path.abspath(image_data_path)
    augmented_data_path = os.path.abspath(augmented_data_path)
    output_path = os.path.abspath(output_path)

    image_paths = [i for i in glob.iglob(os.path.join(image_data_path, "*.png"))]
    print(len(image_paths))
    # Create folders if not exist
    os.makedirs(augmented_data_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Delete bottlenecks file
    if remove_bottleneck:
        delete_bottleneck()

    # Divide dataset
    train_images, validation_images, test_images = divide_data_set(image_paths, percentage_validation=0.1, percentage_test=0.1)

    for imageset_paths, fileset_name in zip([train_images, validation_images, test_images],['training', 'validation', 'test']):

        # Augment data
        augmented_images_paths = start_pipeline(imageset_paths, augmented_data_path, (int(images_height), int(images_width)))

        # Concatenate annotations
        annotations = []
        for augmented_image_path in augmented_images_paths:
            annotations.append(get_formatted_box_string_for(augmented_image_path))

        # Write the annotations file
        print('Save annotations file to ' + output_path + '/' + fileset_name + '.txt')
        save_file(output_path + '/' + fileset_name + '.txt', annotations)

        # Calculate and save anchors
        annotation_dims = []

        size = np.zeros((1, 1, 3))
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
        num_clusters = 0  # NOTE: Doors are rarely overlapping

        if num_clusters == 0:
            for num_clusters in range(1, 7):  # we make 1 through 10 clusters
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

    # Copy classes and anchor files
    shutil.copy2(image_data_path + "/classes.txt", output_path + "/classes.txt")
    shutil.copy2(image_data_path + "/anchors.txt", output_path + "/anchors.txt")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare the training data for YOLO training.')
    parser.add_argument('-i', '--input_path', default='raw_data', type=str, help='Path to your raw image data and bounding boxes')
    parser.add_argument('-o', '--output_path', default='training_data', help='Path where augmented images will be saved')
    parser.add_argument('-b', '--bottleneck', action="store_true", help='Removes the bottleneck file')
    args = parser.parse_args()

    main(args.input_path, args.output_path, 'dist', args.bottleneck)
