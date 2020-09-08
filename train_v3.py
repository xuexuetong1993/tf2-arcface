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
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)
    for key in cfg.keys():
        print("[*] {} : {}".format(key, cfg[key]))

    if cfg['train_dataset']:
        logging.info("load dataset from:{}".format(cfg['train_dataset']))
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'],cfg["input_size"] ,cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])

    logdir = os.path.join('./logs', cfg['sub_name'])
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "metrics"), flush_millis=10000)
    summary_writer.set_as_default()
    
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
                             training=True,
                             use_pretrain=cfg['imagenet_pretrain'])
        parallel_model.summary(line_length=120)

        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['base_lr'], momentum=0.9)

        #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=parallel_model)
        global_step = optimizer.iterations
        global_step_value = global_step.numpy()
        max_iter = cfg["epochs"] * steps_per_epoch
        initial_lr = cfg['base_lr']

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                #return optimizer.lr
                return optimizer.learning_rate
            return lr
        lr_metric = get_lr_metric(optimizer)
        loss_fn = SoftmaxLoss()

        parallel_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc', lr_metric])

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
            print("[*] training from scratch.")
            epochs, steps = 1, 1


    print("---------------fit mode ----------------")


    save_flag = "margin_{}_scale_{}_lr_{}_size_{}".format(cfg["margin"], cfg["logist_scale"],
                            cfg["base_lr"], cfg["input_size"])
    keras_checkpoint_path = "Checkpoints/{}/{}".format(cfg['sub_name'], save_flag + "_Epoch_{epoch}")
    print("ckpt path: ", keras_checkpoint_path)
    parallel_model.save_weights(keras_checkpoint_path)
    def lr_cosin(epoch):
        lr = 0.5 * cfg["base_lr"] * (1.0 + tf.math.cos(3.141592654 * epoch / cfg['epochs']))
        tf.summary.scalar("learning rate", data = lr, step = epoch)
        return lr 
    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    
    tf.keras.callbacks.ModelCheckpoint(filepath=keras_checkpoint_path, save_freq="epoch", save_weights_only=True),

    tf.keras.callbacks.LearningRateScheduler(lr_cosin)
    ]

    #steps_per_epoch = 2
    print("      Start training ....................")
    parallel_model.fit(train_dataset,
              epochs=cfg['epochs'],
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              initial_epoch=epochs - 1)

    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
