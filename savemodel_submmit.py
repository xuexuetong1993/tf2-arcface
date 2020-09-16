from absl import app, flags, logging
import math
from absl.flags import FLAGS
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices, True)
    print("set_memory_growth")
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.dataset as dataset

flags.DEFINE_string('cfg_path', ' ', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'save', ['fit', 'eager_tf', "save"],
                  'fit: model.fit, eager_tf: custom GradientTape')

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    #set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    print("----------params:--------\n", cfg)
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
    parallel_model.summary(line_length=120)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['base_lr'], momentum=0.9, nesterov=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=parallel_model)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['base_lr'], momentum=0.9)

    checkpoint_dir = cfg['checkpoint_dir']
    if checkpoint_dir is not None:
            print("[*] load ckpt from {}".format(checkpoint_dir))
            #parallel_model.load_weights(ckpt_path)
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("load is succed") 
    else:
        print("[*] load ckpt failed.")
        os._exit(0)

    #parallel_model = multi_gpu_model(model, gpus=[0,1,2,3])
    print("---------------fit mode ----------------")
    epoch = 2
    save_flag = "margin_{}_scale_{}_lr_{}_size_{}_epoch_{}".format(cfg["margin"], cfg["logist_scale"], cfg["base_lr"], cfg['input_size'], epoch)
    SavedModel_path = "SavedModel/{}/{}".format(cfg['sub_name'], save_flag)

    tf.saved_model.save(parallel_model, SavedModel_path)
    print("[*] SavedModel done!")


if __name__ == '__main__':
    app.run(main)
