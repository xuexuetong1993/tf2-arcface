from absl import app, flags, logging
import math
from absl.flags import FLAGS
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model
from modules.models import ArcFaceModel, AttModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.dataset as dataset

flags.DEFINE_string('cfg_path', ' ', 'config file path')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')


class _ExtractModule_cover(tf.Module):
    def __init__(self, savedmodel):
        self.model = savedmodel 

    
    def process_tf_efn(self, inputs):
        img_resize = tf.image.resize(inputs, [469, 469])
        crop_img = tf.image.resize_with_crop_or_pad(img_resize, 448, 448)
        return crop_img

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image') ])
    def ExtractFeatures(self, input_image):    
        image_tensor = tf.cast(input_image, tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0, name='image/expand_dims')
        image_tensor = self.process_tf_efn(image_tensor)

        extracted_features = self.model(image_tensor)
        print("extracted_features:", extracted_features)
        extracted_features = tf.reshape(extracted_features, [512,])
        output_global = tf.nn.l2_normalize(extracted_features, axis=0, name='l2_normalization')

        print("output_global: ", output_global)
        named_output_tensors = {}
        named_output_tensors['global_descriptor'] = tf.identity(output_global, name='global_descriptor')
        return named_output_tensors
def main(_):

    cfg = load_yaml(FLAGS.cfg_path)
    for key in cfg.keys():
        print("[*] {} : {}".format(key, cfg[key]))
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        if cfg['att_or_arcface'] == "arcface":

            parallel_model = ArcFaceModel(size=cfg['input_size'],
                             backbone_type=cfg['backbone_type'],
                             num_classes=cfg['num_classes'],
                             head_type=cfg['head_type'],
                             embd_shape=cfg['embd_shape'],
                             w_decay=cfg['w_decay'],
                             margin=float(cfg['margin']),
                             logist_scale=float(cfg['logist_scale']), 
                             training=False,
                             use_pretrain=False)
        elif cfg["att_or_arcface"] == "att":
            
            parallel_model = AttModel(size=cfg['input_size'],
                             backbone_type=cfg['backbone_type'],
                             num_classes=cfg['num_classes'],
                             head_type=cfg['head_type'],
                             embd_shape=cfg['embd_shape'],
                             w_decay=cfg['w_decay'],
                             margin=float(cfg['margin']),
                             logist_scale=float(cfg['logist_scale']), 
                             training=False,
                             use_pretrain=False)
        
        else:
            print("[----] wrong cfg params for model type, check att_or_arcface")
            os._exit(0)
        parallel_model.summary(line_length=160)

        ckpt_path = cfg["checkpoint_dir"]
        print("ckpt_path:", ckpt_path)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            print("[*] load ckpt from {}".format(ckpt_path))
            latest = tf.train.latest_checkpoint(ckpt_path)
            parallel_model.load_weights(latest)
            print("[*] load {} is succed!".format(latest))
            epochs, steps = 1, 1
        else:
            print("[*] {} path is wrong, check if exists!".format(ckpt_path))
            epochs, steps = 1, 1

    print("---------------savedmodel start ----------------")
    #tf.saved_model.save(parallel_model, save_path)

    def conver_submmit(savedmodel, model_save_path):

        #if not os.path.exists(out_path):
        #    os.makedirs(out_path)

        module = _ExtractModule_cover(savedmodel)
        served_function = module.ExtractFeatures
        tf.saved_model.save(module, model_save_path, signatures={'serving_default': served_function})

        print("save to : {}".format(model_save_path))
        print("[*] job well done!")



    save_path =  os.path.join("SavedModel/", cfg["saved_model_name"])
    conver_submmit(parallel_model, save_path)


if __name__ == '__main__':
    app.run(main)
