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

        # apply offset to to have zero-indexed ground truth classes
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[box_np.shape[0]], dtype=np.int32) - params.settings.label_id_offset)

        # do one-hot encoding to ground truth classes
        classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, params.settings.num_classes))

    print('Done prepping data.')
    
    return image_tensors, classes_one_hot_tensors, box_tensors




def save_detections_from_folder(image_dir, category_index, detection_model):
    image_dir = Path(image_dir)
    images_np = []
    for img_path in image_dir.glob("*.jpg"):
        print(img_path)
        images_np.append(np.expand_dims(
          draw_utils.load_image_into_numpy_array(img_path), axis=0))
    
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
        
        print(f"detected_image_{i+1}.jpg")
        if not os.path.exists('./media'): os.mkdir('./media')
        if not os.path.exists('./media/out'): os.mkdir('./media/out')
        plt.imsave(f"./media/out/detected_image_{i+1}.jpg", images_np[i][0])