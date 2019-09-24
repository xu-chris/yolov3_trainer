import cv2
import os
import numpy as np


def rand(a=0, b=1):
    return np.random.rand() * (b-a) + a


def check_file_for(image_path):
    # Get path to yolo txt file
    path, full_file_name = os.path.split(image_path)
    file_name, _ = os.path.splitext(full_file_name)
    bounding_box_file = path + '/' + file_name + ".txt"

    if not os.path.isfile(bounding_box_file):
        return False
    return True


def load_files(image_path):

    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Get path to yolo txt file
    path, full_file_name = os.path.split(image_path)
    file_name, _ = os.path.splitext(full_file_name)
    bounding_box_file = path + '/' + file_name + ".txt"

    file = open(bounding_box_file, "r")

    bounding_boxes = []
    if file.mode == 'r':
        lines = file.readlines()
        for line, index in zip(lines, range(len(lines))):
            lines[index] = line.rstrip("\n")
            bounding_boxes.append([float(i) for i in line.split(" ")])

    file.close()

    return [image], [bounding_boxes]


def resize(images, bounding_boxes_lists, size=(608, 608)):

    resized_images = []

    for image in images:
        resized_images.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))

    return resized_images, bounding_boxes_lists


def crop(images, bounding_boxes_lists, save_initial_images=False):

    run_once = False
    for image, bounding_boxes in zip(images, bounding_boxes_lists):

        current_mode = 0

        # Crop it based on the mode. We have three modes: Left, Center and Right
        for mode in [0, 1, 2]:

            # Get the min side length
            height, width, _ = image.shape
            min_side_length = min(width, height)

            # Decide the leading and trailing boundaries

            if mode == 0:
                leading_boundary = 0
                trailing_boundary = min_side_length
            elif mode == 1:
                leading_boundary = (width / 2) - (min_side_length / 2)
                trailing_boundary = (width / 2) + (min_side_length / 2)
            else:
                leading_boundary = width - min_side_length
                trailing_boundary = width

            # Safely convert them after calculation
            leading_boundary = int(leading_boundary)
            trailing_boundary = int(trailing_boundary)

            # Crop bounding boxes and adjust their sizes
            leading_boundary_norm = leading_boundary / width
            trailing_boundary_norm = trailing_boundary / width

            new_bounding_boxes = []

            for bounding_box in bounding_boxes:

                new_bounding_box = bounding_box.copy()

                # Skip when center is outside of the boundaries
                if bounding_box[1] < leading_boundary_norm or bounding_box[1] > trailing_boundary_norm:
                    continue
                # Adjust X of bounding box
                new_bounding_box[1] = ((new_bounding_box[1] * width) - leading_boundary) / min_side_length
                # Adjust width of bounding box
                new_bounding_box[3] = new_bounding_box[3] * (width / min_side_length)

                new_bounding_boxes.append(new_bounding_box)

            # Continue with the next crop mode if there is no bounding box
            if new_bounding_boxes is []:
                current_mode += 1
                continue

            # Crop image
            new_image = image[:, leading_boundary:trailing_boundary]

            # Decide to save the initial images or just drop them
            if not run_once and not save_initial_images:
                images = [new_image]
                bounding_boxes_lists = [new_bounding_boxes]
            else:
                images.append(new_image)
                bounding_boxes_lists.append(new_bounding_boxes)

            run_once = True
            current_mode += 1
    return images, bounding_boxes_lists


def flip_vertical(images, bounding_boxes_to_images):

    for image, bounding_boxes, index in zip(images, bounding_boxes_to_images, range(len(images))):

        image = image[:, ::-1]
        new_bounding_boxes = []

        for bounding_box in bounding_boxes:
            new_bounding_box = bounding_box.copy()
            new_bounding_box[1] = -(bounding_box[1]-0.5)+0.5
            new_bounding_boxes.append(new_bounding_box)

        images.append(image)
        bounding_boxes_to_images.append(new_bounding_boxes)

    return images, bounding_boxes_to_images


def change_hsv(images, bounding_boxes_lists, hsv_images = 5):

    for image, bounding_boxes, index in zip(images, bounding_boxes_lists, range(len(images))):

        for image_round in range(hsv_images):

            # Get HSV image
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Get current HSV values
            H = img_hsv[:, :, 0].astype(np.float32)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            # Modify Hue
            hue = rand(-0.1, 0.1) * 255
            H += hue
            np.clip(H, a_min=0, a_max=255, out=H)

            # Modify Sat
            sat = rand(1, 1.5) if rand() < 0.5 else 1 / rand(1, 1.5)
            S *= sat
            np.clip(S, a_min=0, a_max=255, out=S)

            # Modify Val
            val = rand(1, 1.5) if rand() < 0.5 else 1 / rand(1, 1.5)
            V *= val
            np.clip(V, a_min=0, a_max=255, out=V)

            # Concatenate dimensions together again
            img_hsv[:, :, 0] = H.astype(np.uint8)
            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)

            # Return
            images.append(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))
            bounding_boxes_lists.append(bounding_boxes)

    return images, bounding_boxes_lists


def save_files(images, bounding_boxes_lists, image_path, output_path):

    # Get image name to add additional names to it
    _, full_file_name = os.path.split(image_path)
    file_name, file_extension = os.path.splitext(full_file_name)

    paths = []

    for image, bounding_boxes, index in zip(images, bounding_boxes_lists, range(len(images))):

        # cv2.imshow('image', image)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

        file_path = "{}/{}_{}".format(output_path,index, file_name)

        full_path = file_path + file_extension

        # Save image
        cv2.imwrite(full_path, image)

        # Save bounding boxes
        file = open(file_path + '.txt', "w+")
        for bounding_box in bounding_boxes:
            # Skip empty lines
            if bounding_box is None:
                continue
            file.write("{} {} {} {} {}\n".format(int(bounding_box[0]), bounding_box[1], bounding_box[2],
                                                 bounding_box[3],
                                                 bounding_box[4]))
        file.close()
        paths.append(file_path + '.png')

    return paths
