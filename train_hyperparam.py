"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import os
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from datetime import datetime


def _main():

    # Set train variables
    training_set_annotation_path = 'dist/training.txt'
    validation_set_annotation_path = 'dist/validation.txt'
    test_set_annotation_path = 'dist/test.txt'
    classes_path = 'dist/classes.txt'
    pretrained_weights_path = 'source/darknet53_weights.h5'
    logs_root_path = 'logs'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    final_output_path = 'dist/'
    val_split = 0.1

    # Split training and validation set
    with open(training_set_annotation_path) as f:
        training_lines = f.readlines()
    with open(validation_set_annotation_path) as f:
        validation_lines = f.readlines()
    with open(test_set_annotation_path) as f:
        test_lines = f.readlines()

    # Set input size
    input_shape = (608, 608)  # multiple of 32, hw

    # Train with frozen layers first, to get a stable loss.

    batch_sizes = [8, 16, 32]
    init_learning_rates = [1e-2, 1e-3, 1e-4]
    anchors_paths = ['source/yolo_anchors.txt']

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    # Make log_dir folder
    folder_hash = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(logs_root_path, exist_ok=True)
    os.makedirs(logs_root_path + '/' + folder_hash, exist_ok=True)
    log_dir = logs_root_path + '/' + folder_hash + '/'

    # set logging
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    for batch_size in batch_sizes:

        os.makedirs(logs_root_path + '/' + folder_hash + '/' + 'batchsize-' + str(batch_size), exist_ok=True)

        for init_learning_rate in init_learning_rates:

            os.makedirs(logs_root_path + '/' + folder_hash + '/' + 'batchsize-' + str(batch_size) + '/' + 'lr-' + str(init_learning_rate), exist_ok=True)

            for anchors_path in anchors_paths:
                anchors = get_anchors(anchors_path)

                anchors_filename = os.path.splitext(os.path.basename(anchors_path))[0]

                # Make log dir
                os.makedirs(logs_root_path + '/' + folder_hash + '/' + 'batchsize-' + str(batch_size) + '/' + 'lr-' + str(init_learning_rate) + '/' + anchors_filename, exist_ok=True)
                log_dir = logs_root_path + '/' + folder_hash + '/' + 'batchsize-' + str(batch_size) + '/' + 'lr-' + str(init_learning_rate) + '/' + anchors_filename + '/'

                # set logging
                logging = TensorBoard(log_dir=log_dir, update_freq='batch')

                # Load and create model
                model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=pretrained_weights_path)

                # use custom yolo_loss Lambda layer.
                model.compile(optimizer=Adam(lr=init_learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                print('-----------------------------------------------------------------------------------------')
                print('Train on {} samples, val on {} samples, with batch size {} and learning rate {} with anchor file: {}'.format(len(training_lines), len(validation_lines), batch_size, init_learning_rate, anchors_path))
                print('-----------------------------------------------------------------------------------------')
                model.fit_generator(
                    data_generator_wrapper(training_lines, batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, len(training_lines)//batch_size),
                    validation_data=data_generator_wrapper(validation_lines, batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, len(validation_lines)//batch_size),
                    epochs=15,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint, reduce_lr]
                )
                model.save_weights(log_dir + 'test__weights_batch-size-' + str(batch_size) + '_lr-' + str(init_learning_rate) + '.h5')
                model.save(log_dir + 'test__models_batch-size-' + str(batch_size) + '_lr-' + str(init_learning_rate) + '.h5')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='source/yolo_weights.h5'):
    '''create the training model'''

    # get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
