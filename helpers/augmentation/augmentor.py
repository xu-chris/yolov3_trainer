from . import augmentation_methods as am
import glob
import os
import math


def pipeline(image_path, output_path, size=(608, 608)):
    processes = 6
    step = 1
    if not am.check_file_for(image_path):
        print('No bounding box file found for: {}'.format(image_path))
        return []

    # 1
    print('({}{}) File: {}, loading image...                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\r')
    images, bounding_boxes_list = am.load_files(image_path)
    step += 1

    # # 2
    # print('({}{}) File: {}, cropping image...                                 '.format(
    #     '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    # ), end='\r')
    # images, bounding_boxes_list = am.crop(images, bounding_boxes_list)
    # step += 1

    # 3
    print('({}{}) File: {}, resizing image...                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\r')
    images, bounding_boxes_list = am.resize(images, bounding_boxes_list, size)
    step += 1

    # 4
    print('({}{}) File: {}, flipping image...                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\r')
    images, bounding_boxes_list = am.flip_vertical(images, bounding_boxes_list)
    step += 1

    # 5
    print('({}{}) File: {}, change HSV...                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\r')
    images, bounding_boxes_list = am.change_hsv(images, bounding_boxes_list, hsv_images=5)
    step += 1

    # 6
    print('({}{}) File: {}, saving image...                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\r')
    paths = am.save_files(images, bounding_boxes_list, image_path, output_path)
    step += 1

    # 7
    print('({}{}) File: {}, Done!                                 '.format(
        '█' * math.floor(step), '░' * math.ceil(processes - step), image_path
    ), end='\n')

    return paths


def start_pipeline(image_paths, output_path='output_images', size=(608, 608)):
    # find out all images in the folder
    print('Start augmenting images. Output path: {}'.format(output_path))
    iterator = 1
    all_paths = []
    visual_steps = 20

    for image_path in image_paths:
        # Abspath
        image_path = os.path.abspath(image_path)
        print('Start augmenting image {}. Input path: {}'.format(iterator, image_path))
        all_paths.extend(pipeline(image_path, output_path, size))
        iterator += 1

    print('Finished augmenting images')

    return all_paths


if __name__ == '__main__':

    path = 'images/'
    paths = glob.iglob(os.path.join(path, "*.png"))
    start_pipeline(paths)
