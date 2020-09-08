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
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
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
    if cfg['train_dataset']:
        logging.info("load gldv2 clean dataset from:{}".format(cfg['train_dataset']))
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'],cfg["input_size"] ,cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])

    #initial_learning_rate = tf.constant(cfg['base_lr'])

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
                             use_pretrain=True)
        parallel_model.summary(line_length=120)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=cfg['base_lr'], momentum=0.9, nesterov=True)
        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr
        lr_metric = get_lr_metric(optimizer)
        #lr_decayed_fn = tf.keras.experimental.CosineDecay(
        #    initial_learning_rate=cfg['base_lr'], decay_steps=steps_per_epoch)
        loss_fn = SoftmaxLoss()
        parallel_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc', lr_metric])
        #para_model = multi_gpu_model(model, gpus=8)

        ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            parallel_model.load_weights(ckpt_path)
            epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
        else:
            print("[*] training from scratch.")
            epochs, steps = 1, 1

    #parallel_model = multi_gpu_model(model, gpus=[0,1,2,3])
    print("---------------fit mode ----------------")

    """
    mc_callback = ModelCheckpoint(
        'checkpoints/' + cfg['sub_name'] + '/e_{epoch}_b_.ckpt',
        save_freq="epoch", verbose=1)
    tb_callback = TensorBoard(log_dir='logs/',
                              update_freq=cfg['batch_size'] * 5,
                              profile_batch=0)
    tb_callback._total_batches_seen = steps
    tb_callback._samples_seen = steps * cfg['batch_size']
    callbacks = [mc_callback, tb_callback, lr_callback]
    """
    class LearningRateReducerCb(tf.keras.callbacks.Callback):

        """
        def lr_cosin(epoch, lr):
            max_epoch = cfg['epochs'] 
            base_lr = lr
            return 0.5 * base_lr * (1.0 + tf.math.cos(3.1415926 * epoch / max_epoch))
        """
        def on_epoch_end(self, epoch, logs={}):
            old_lr = parallel_model.optimizer.lr.numpy()
            base_lr = cfg["base_lr"]
            max_epoch = cfg['epochs'] 
            new_lr = 0.5 * base_lr * (1.0 + tf.math.cos(3.1415926 * epoch / max_epoch))
            print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
            tf.keras.backend.set_value(parallel_model.optimizer.lr, new_lr)
            #parallel_model.optimizer.lr.assign(new_lr)

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {} \n'.format(epoch + 1, parallel_model.optimizer.lr))

    save_flag = "margin_{}_scale_{}_lr_{}_size_{}".format(cfg["margin"], cfg["logist_scale"],
                            cfg["base_lr"], cfg["input_size"])
    keras_checkpoint_path = "Checkpoints/{}/{}".format(cfg['sub_name'], save_flag)

    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    
    tf.keras.callbacks.ModelCheckpoint(filepath=keras_checkpoint_path,save_freq="epoch",
                                   save_weights_only=True)
    #LearningRateReducerCb()
    ]

    #steps_per_epoch = 2
    print("start training")
    parallel_model.fit(train_dataset,
              epochs=cfg['epochs'],
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              initial_epoch=epochs - 1)

    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
