from dataclasses import dataclass
import tensorflow as tf

@dataclass
class Hyperparameters:
    batch_size: int
    num_batches: int
    optimizer: any
    min_score_thresh: float


hyperparameters = Hyperparameters(batch_size=7,
                            num_batches=100,
                            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                            min_score_thresh=0.8)


@dataclass
class Settings:
    num_classes: int
    case_class_id: int
    label_id_offset: int
    class_name: str
    pipeline_config: str
    checkpoint_path: str


settings = Settings(num_classes=1,
                   case_class_id=1,
                   label_id_offset=1,
                   class_name='case',
                   pipeline_config='./models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config',
                   checkpoint_path='./models/research/object_detection/test_data/checkpoint/ckpt-0')


category_index = {settings.case_class_id: {'id': settings.case_class_id,
                                  'name': settings.class_name}}