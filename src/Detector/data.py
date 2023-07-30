from pathlib import Path
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from object_detection.utils import visualization_utils as viz_utils

import Utilities.drawing_utilities as draw_utils
import Program.parameters as params
import Detector.model as model

def process_training_folder(image_dir):

    '''Process the selected folder and load images with boxes into separate lists of numpy arrays.

    Looks up all .xml files in the given directory and parse them
    to find image names and their box coordinates.
    Then load images with their boxes into lists of arrays.

    Args:
    image_dir: a directory path

    Returns:
    images_np: a list of float numpy arrays with shape (img_height, img_width, 3)
    boxes_np: a list of float numpy arrays that represent the boxes
    '''

    image_dir = Path(image_dir)
    images_np = []
    boxes_np = []

    for xml_path in image_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_path = root.find('path').text
        x_min = np.int32(root.find('object').find('bndbox').find('xmin').text)/np.int32(root.find('size').find('width').text)
        y_min = np.int32(root.find('object').find('bndbox').find('ymin').text)/np.int32(root.find('size').find('height').text)
        x_max = np.int32(root.find('object').find('bndbox').find('xmax').text)/np.int32(root.find('size').find('width').text)
        y_max = np.int32(root.find('object').find('bndbox').find('ymax').text)/np.int32(root.find('size').find('height').text)

        images_np.append(draw_utils.load_image_into_numpy_array(Path(img_path)))
        boxes_np.append(np.array([[y_min, x_min, y_max, x_max]], dtype=np.float32))

        print("Path:", img_path)
        print("Name:", root.find('object').find('name').text)
        print("x_min:", x_min)
        print("y_min:", y_min)
        print("x_max:", x_max)
        print("y_max:", y_max)
        print('\n')
        
    return images_np, boxes_np


def prepping_data(images_np: list, boxes_np: list):
    
    '''Do some data preprocessing before it is fed to the model
    
    Convert the class labels to one-hot representations.
    Convert train images, gt boxes and class labels to tensors.

    Args:
    images_np: a list of images (numpy arrays with shape (img_height, img_width, 3))
    boxes_np:  a list of boxes (also numpy arrays)

    Returns:
    image_tensors: a list of tensors of shape (1, img_height, img_width, 3)
    classes_one_hot_tensors: a list of one-hot representations of class labels
    box_tensors: a list of box tensors
    '''

    image_tensors = []

    # lists containing the one-hot encoded classes and ground truth boxes
    classes_one_hot_tensors = []
    box_tensors = []

    for (image_np, box_np) in zip(images_np, boxes_np):

        # convert training image to tensor, add batch dimension, and add to list
        image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            image_np, dtype=tf.float32), axis=0))

        # convert numpy array to tensor, then add to list
        box_tensors.append(tf.convert_to_tensor(box_np, dtype=tf.float32))

        # apply offset to have zero-indexed ground truth classes
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[box_np.shape[0]], dtype=np.int32) - params.settings.label_id_offset)

        # do one-hot encoding to ground truth classes
        classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, params.settings.num_classes))

    print('Done prepping data.')
    
    return image_tensors, classes_one_hot_tensors, box_tensors




def save_detections_from_folder(image_dir, category_index, detection_model):

    '''Load images from the selected folder, make detections and save them
    
    Gets images from the selected folder, make detections on them,
    then visualize them using special utilities and save to the another folder.

    Args:
    image_dir: a directory path
    category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices. 
    detection_model: a trained model that is ready to make detections

    Returns: nothing.
    '''

    image_dir = Path(image_dir)
    images_np = []

    # load images from the selected folder
    for img_path in image_dir.glob("*.jpg"):
        print(img_path)
        images_np.append(np.expand_dims(
          draw_utils.load_image_into_numpy_array(img_path), axis=0))
    
    # make detections and visualize them on the images
    for i in range(len(images_np)):
        input_tensor = tf.convert_to_tensor(images_np[i], dtype=tf.float32)
        detections = model.detect(input_tensor, detection_model)
        viz_utils.visualize_boxes_and_labels_on_image_array(
                images_np[i][0],
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.uint32) + params.settings.label_id_offset,
                detections['detection_scores'][0].numpy(),
                category_index=category_index,
                use_normalized_coordinates=True,
                min_score_thresh=params.hyperparameters.min_score_thresh)
        
        # save detected images to the another folder
        print(f"detected_image_{i+1}.jpg")
        if not os.path.exists('./media'): os.mkdir('./media')
        if not os.path.exists('./media/out'): os.mkdir('./media/out')
        plt.imsave(f"./media/out/detected_image_{i+1}.jpg", images_np[i][0])