"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import tensorflow as tf
import os
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from datetime import datetime


def _main():
    # Set train variables
    training_set_annotation_path = 'dist/training.txt'
    validation_set_annotation_path = 'dist/validation.txt'
    test_set_annotation_path = 'dist/test.txt'
    classes_path = 'dist/classes.txt'
    source_weights_path = 'source/weights.h5'
    logs_root_path = 'logs'
    final_output_path = 'dist/'
    anchors_path = 'source/yolo_anchors.txt'
    val_split = 0.1

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # Make log_dir folder
    folder_hash = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(logs_root_path, exist_ok=True)
    os.makedirs(logs_root_path + '/' + folder_hash, exist_ok=True)
    log_dir = logs_root_path + '/' + folder_hash + '/'

    # set logging
    logging = TensorBoard(log_dir=log_dir, update_freq='batch')
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    # Split training and validation set
    with open(training_set_annotation_path) as f:
        training_lines = f.readlines()
    with open(validation_set_annotation_path) as f:
        validation_lines = f.readlines()
    with open(test_set_annotation_path) as f:
        test_lines = f.readlines()

    # Set input size
    input_shape = (608, 608)  # multiple of 32, hw

    # Create model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.as_default()
    # (nothing gets printed in Jupyter, only if you run it standalone)
    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=source_weights_path)
    model.summary()

    # Train with frozen layers first, to get a stable loss.
    if True:
        batch_size = 16
        learning_rate = 1e-3

        # use custom yolo_loss Lambda layer.
        model.compile(optimizer=Adam(lr=learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('-----------------------------------------------------------------------------------------')
        print('Train on {} samples, val on {} samples, with batch size {} and learning rate {} with anchor file: {}'.format(len(training_lines), len(validation_lines), batch_size, learning_rate, anchors_path))

        model.fit_generator(
            data_generator_wrapper(training_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, len(training_lines) // batch_size),
            validation_data=data_generator_wrapper(validation_lines, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, len(validation_lines) // batch_size),
            epochs=10,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr]
        )
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        model.save(log_dir + 'trained_model_stage_1.h5')

    # Unfreeze model and train for 50 epochs
    if True:

        batch_size = 16
        learning_rate = 1e-4

        # Unfreeze layers
        print('Unfreeze all of the layers for shocking.')
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        model.compile(optimizer=Adam(lr=learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

        print('-----------------------------------------------------------------------------------------')
        print('Train on {} samples, val on {} samples, with batch size {} and learning rate {} with anchor file: {}'.format(len(training_lines), len(validation_lines), batch_size, learning_rate, anchors_path))
        model.fit_generator(
            data_generator_wrapper(training_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, len(training_lines) // batch_size),
            validation_data=data_generator_wrapper(validation_lines, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, len(validation_lines) // batch_size),
            epochs=60,
            initial_epoch=10,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping]
        )
        model.save_weights(log_dir + 'trained_weights_final.h5')
        model.save(log_dir + 'trained_model_final.h5')
        model.save_weights(final_output_path + 'trained_weights_final.h5')
        model.save(final_output_path + 'trained_model_final.h5')


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
