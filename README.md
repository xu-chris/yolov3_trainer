# YOLO v3 Trainer

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/210558245.svg)](https://zenodo.org/badge/latestdoi/210558245)

This repository is based on [@qqwweee's implementation of YOLO v3](https://github.com/qqwweee/keras-yolo3) and the very useful commits by [@KUASWoodyLIN](https://github.com/qqwweee/keras-yolo3/pull/206) who has pushed the evaluation of the CNN net to it's boundaries.

The purpose of the trainer repository is to split the given implementation by @qqwweee into two packages:
- Training
- [Detection](https://github.com/creichel/yolov3_detector)

This package is about the training of YOLOv3 and can be easily extended to make the training even more better or to extend the output of the training evaluation. Because the resulting model can be better thanks to better augmentation scripts, but the actual use of the model will be still the detector's job to do. Thus, we can use the same detector implementation for different models (even for improved ones) without updating the package itself.

Another benefit is the smaller size for the deployment of the model itself.

## Getting started

To train your own detector, you have to go through multiple steps:

1. Record or find multiple images of the object you want to detect.
    Simply use a camera or use any of the image sets available in the internet.

    | **Note:** be aware of if you're taking pictures by yourself, the attributes of the camera (sensor, perspective, image size, etc.) as well as of your environment might have a huge impact of the generalization of your detector! |
    | --- |

2. Label images.
    Use [labelImg](https://github.com/tzutalin/labelImg) to annotate the images with bounding box information. This creates to every image an `.txt` file with the bounding box information and the class label.

    | **Note:** Save the annotation with YOLO style |
    | --- |

3. Copy the images and txt files in `raw_data` folder.
    For the next step, the images and txt files are read from this folder.

4. Run preparation script.
    Check the settings of `prepare_training.py`. This script does the following things:
    - Annotate the dataset with flipped images with different color settings
    - Save the annotated images into `training_data` folder
    - Divide the set of images into three subsets, saved into `dist` folder:
        - `training.txt` for training
        - `validation.txt` for the validation of the set during training
        - `test.txt` for keeping a set of images for the later evaluation of the detector itself

        These files are already formatted as the following:   
        - One row for one image
            - Row format: `image_file_path box1 box2 ... boxN`;  
            - Box format: `x_min,y_min,x_max,y_max,class_id` (no space).    

        Here is an example:
        ```
        path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
        path/to/img2.jpg 120,300,250,600,2
        ...
        ```
    - Create or copy `classes.txt` and `anchors.txt` (if you train the whole network from the beginning, use the created `anchors.txt`. Otherwise copy the file from the model you will use)

5. Copy the pretrained network

    ```
    wget https://pjreddie.com/media/files/yolov3.weights
    ```

    For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

    If you want to use original pretrained weights for YOLOv3:  
    ```
    wget https://pjreddie.com/media/files/darknet53.conv.74
   ```

6. Run `python convert.py -w yolov3.cfg yolov3.weights source/weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

7. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.
---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

## Attribution
 Thanks again to [@qqwweee's implementation of YOLO v3](https://github.com/qqwweee/keras-yolo3), the useful commits by [@KUASWoodyLIN](https://github.com/qqwweee/keras-yolo3/pull/206) and of course [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)'s implementation that inspires the implementation of YOLO v3 in Keras.
