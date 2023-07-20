import tensorflow as tf
import random
from object_detection.utils import config_util
from object_detection.builders import model_builder

import Program.parameters as params

def prepare_model_for_training(pipeline_config: str, checkpoint_path: str):
    
    tf.keras.backend.clear_session()
    
    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    
    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']
    
    # Modify the number of classes from its default of 90
    model_config.ssd.num_classes = params.settings.num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True
    
    #Model Builder
    detection_model = model_builder.build(model_config, is_training=True)
    
    #Define checkpoints for the box predictor
    tmp_box_predictor_checkpoint = tf.train.Checkpoint(
        _base_tower_layers_for_heads = detection_model._box_predictor._base_tower_layers_for_heads,
        _box_prediction_head = detection_model._box_predictor._box_prediction_head)
    
    #Define the temporary model checkpoint
    tmp_model_checkpoint = tf.train.Checkpoint(
        _feature_extractor = detection_model._feature_extractor,
        _box_predictor = tmp_box_predictor_checkpoint)
    
    # Define a checkpoint that sets `model` to the temporary model checkpoint
    checkpoint = tf.train.Checkpoint(
        model = tmp_model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(checkpoint_path)
    
    #Run a dummy image to generate the model variables

    # use the detection model's `preprocess()` method and pass a dummy image
    tmp_image, tmp_shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    tmp_detections = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)
    
    tf.keras.backend.set_learning_phase(True)
    
    to_fine_tune = []
    for v in detection_model.trainable_variables:
        if v.name.startswith(('WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
                            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead')):
            to_fine_tune.append(v)

    detection_model.provide_groundtruth(groundtruth_boxes_list=[], groundtruth_classes_list=[])

    print('Weights restored!')
    
    return detection_model, to_fine_tune



@tf.function
def train_step_fn(image_list,
                groundtruth_boxes_list,
                groundtruth_classes_list,
                model,
                optimizer,
                vars_to_fine_tune):
    """A single training iteration.

    Args:
      image_list: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """

    with tf.GradientTape() as tape:
        # Preprocess the images

        preprocessed_image_list = []
        true_shape_list = []

        for img in image_list:
            processed_img, true_shape = model.preprocess(img)
            preprocessed_image_list.append(processed_img)
            true_shape_list.append(true_shape)

        preprocessed_image_tensor = tf.concat(preprocessed_image_list, axis=0)
        true_shape_tensor = tf.concat(true_shape_list, axis=0)

        # Make a prediction
        prediction_dict = model.predict(preprocessed_image_tensor, true_shape_tensor)

        # Calculate the total loss (sum of both losses)

        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)

        losses_dict = model.loss(prediction_dict, true_shape_tensor)

        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        # Calculate the gradients
        gradients = tape.gradient(total_loss, vars_to_fine_tune)

        # Optimize the model's selected variables
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

        
    return total_loss



def train_the_model(images_np: list,
                    image_tensors,
                    classes_one_hot_tensors,
                    box_tensors,
                    detection_model,
                    to_fine_tune):

    print('Start fine-tuning!', flush=True)

    for idx in range(params.hyperparameters.num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:params.hyperparameters.batch_size]

        # Get the ground truth
        boxes_list = [box_tensors[key] for key in example_keys]
        classes_list = [classes_one_hot_tensors[key] for key in example_keys]

        # get the images
        tmp_image_tensors = [image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(tmp_image_tensors,
                                   boxes_list,
                                   classes_list,
                                   detection_model,
                                   params.hyperparameters.optimizer,
                                   to_fine_tune
                                  )

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(params.hyperparameters.num_batches)
            + ', loss=' +  str(total_loss.numpy()), flush=True)

    print('Done fine-tuning!')




@tf.function
def detect(input_tensor, detection_model):
    """Run detection on an input image.

    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)

    # use the detection model's postprocess() method to get the the final detections
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections