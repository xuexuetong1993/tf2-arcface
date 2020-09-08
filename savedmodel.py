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
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.dataset as dataset

flags.DEFINE_string('cfg_path', ' ', 'config file path')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

def main(_):

    cfg = load_yaml(FLAGS.cfg_path)
    for key in cfg.keys():
        print("[*] {} : {}".format(key, cfg[key]))
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        parallel_model = ArcFaceModel(size=cfg['input_size'],
                             backbone_type=cfg['backbone_type'],
                             num_classes=cfg['num_classes'],
                             head_type=cfg['head_type'],
                             embd_shape=cfg['embd_shape'],
                             w_decay=cfg['w_decay'],
                             margin=float(cfg['margin']),
                             logist_scale=float(cfg['logist_scale']), 
                             training=False,
                             use_pretrain=False,
                             l2_norm=True)
        parallel_model.summary(line_length=80)

        #optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['base_lr'], momentum=0.9)
        #loss_fn = SoftmaxLoss()

        #parallel_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

        ckpt_path = cfg["checkpoint_dir"]
        print("ckpt_path:", ckpt_path)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            print("[*] load ckpt from {}".format(ckpt_path))
            #checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
            latest = tf.train.latest_checkpoint(ckpt_path)
            parallel_model.load_weights(latest)
            print("[*] load {} is succed!".format(ckpt_path))
            epochs, steps = 1, 1
        else:
            print("[*] {} path is wrong, check if exists!".format(ckpt_path))
            epochs, steps = 1, 1


    print("---------------savedmodel start ----------------")
    save_path =  os.path.join("SavedModel/", cfg["saved_model_name"])
    tf.saved_model.save(parallel_model, save_path)

    print("save to : {}".format(save_path))
    print("[*] job well done!")


if __name__ == '__main__':
    app.run(main)
