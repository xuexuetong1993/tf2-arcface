"""Densenet Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
from tqdm import tqdm
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
import modules.dataset as dataset
import time
import numpy as np
flags.DEFINE_string('cfg_path', ' ', 'config file path')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')


class Train(object):
  """Train class.

  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
    batch_size: Batch size.
    strategy: Distribution strategy in use.
  """

  def __init__(self, optimizer, checkpoint, cfg, epochs, enable_function, model, batch_size, strategy, learning_rate, step_per_epoch):
    self.epochs = epochs
    self.checkpoint = checkpoint
    self.cfg = cfg
    self.batch_size = batch_size
    self.enable_function = enable_function
    self.strategy = strategy
    self.learning_rate = learning_rate
    loss_fn = SoftmaxLoss()
    self.loss_object = loss_fn
    self.optimizer = optimizer
    self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.step_per_epoch = step_per_epoch
    # self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
    #     name='test_accuracy')
    self.model = model
    self.model.summary()
    print(self.cfg['checkpoint_dir'])

  def lr_cosin(self, cur_epoch):
    max_epoch = self.epochs
    base_lr = self.learning_rate
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))

  def compute_loss(self, label, logist):
    loss = tf.reduce_sum(self.loss_object(label, logist)) * (
        1. / self.batch_size)
    loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
    return loss

  def train_step(self, inputs):
    """One train step.

    Args:
      inputs: one batch input.

    Returns:
      loss: Scaled loss.
    """

    images, label = inputs
    with tf.GradientTape() as tape:
      logist = self.model(images, training=True)
      loss = self.compute_loss(label, logist)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))

    self.train_acc_metric(label, logist)
    return loss

  def test_step(self, inputs):
    """One test step.

    Args:
      inputs: one batch input.
    """
    image, label = inputs
    logist = self.model(image, training=False)

    unscaled_test_loss = self.loss_object(label, logist) + sum(
        self.model.losses)

    self.test_acc_metric(label, logist)
    self.test_loss_metric(unscaled_test_loss)

  # def custom_loop(self, train_dist_dataset, test_dist_dataset,
  #                 strategy):
  def custom_loop(self, train_dist_dataset,
                  strategy):
    """Custom training and testing loop.

    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.

    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(ds, epoch, step_per_epoch):
        total_loss = 0.0
        num_train_batches = 0.0
        for one_batch in ds:
            start = time.time()
            per_replica_loss = strategy.run(self.train_step, args=(one_batch,))
            total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            """
            duration = time.time() - start
            if num_train_batches % 10 == 0:   
                template = ('Epoch: {}, Iters: {}/{}, Train Loss: {}, Train Accuracy: {}, Duration: {}')
                print(template.format(epoch, num_train_batches, step_per_epoch,
                      total_loss / num_train_batches,
                      self.train_acc_metric.result(), duration))
            """
            num_train_batches += 1
            #if num_train_batches > step_per_epoch:
             #   break
        return total_loss, num_train_batches

    def distributed_test_epoch(ds):
      num_test_batches = 0.0
      for one_batch in ds:
        strategy.run(self.test_step, args=(one_batch,))
        num_test_batches += 1
      return self.test_loss_metric.result(), num_test_batches

    if self.enable_function:
      distributed_train_epoch = tf.function(distributed_train_epoch)
      # distributed_test_epoch = tf.function(distributed_test_epoch)

    for epoch in range(self.epochs):
        start = time.time()
        self.optimizer.learning_rate = self.lr_cosin(epoch)

        train_total_loss, num_train_batches = distributed_train_epoch(
            train_dist_dataset, epoch, self.step_per_epoch)
        duration = time.time() - start
        # test_total_loss, num_test_batches = distributed_test_epoch(
        #     test_dist_dataset)

        template = ('[Epoch Done] Epoch: {}, Train Loss: {}, Train Accuracy: {}, Duration: {}, Lr: {}')
                    #'Test Loss: {}, Test Accuracy: {}')

        print(
            template.format(epoch,
                            train_total_loss / num_train_batches,
                            self.train_acc_metric.result(), duration, self.lr_cosin(epoch)))
                            # test_total_loss / num_test_batches,
                            # self.test_acc_metric.result()))
        save_flag = "margin_{}_scale_{}_lr_{}_size_{}_acc_{:.2f}".format(self.cfg["margin"], self.cfg["logist_scale"],
                                self.cfg["base_lr"], self.cfg["input_size"],self.train_acc_metric.result())
        keras_model_path = "SavedModel/{}/{}/e_{}".format(self.cfg['sub_name'], save_flag, epoch)
        keras_checkpoint_path = "Checkpoints/{}/{}".format(self.cfg['sub_name'], save_flag)
        self.checkpoint.save(keras_checkpoint_path)
        print("[Succed in save chechpoint]: {}".format(keras_checkpoint_path))
        if epoch % 2 ==0:

            tf.saved_model.save(self.model, keras_model_path)
            print("[Succed in save SavedModel]: {}".format(keras_model_path))

    if epoch != self.epochs - 1:
        self.train_acc_metric.reset_states()
        # self.test_acc_metric.reset_states()

    return (train_total_loss / num_train_batches,
            self.train_acc_metric.result().numpy())
            #test_total_loss / num_test_batches,
            #self.test_acc_metric.result().numpy())

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)


    cfg = load_yaml(FLAGS.cfg_path)
    print("----------params:--------\n", cfg)

    devices = ['/device:GPU:{}'.format(i) for i in range(cfg["num_gpu"])]
    strategy = tf.distribute.MirroredStrategy(devices)

    if cfg['train_dataset']:
        logging.info("load:"+cfg["train_dataset"])
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
                    cfg['train_dataset'], cfg['batch_size'], cfg["input_size"], cfg['binary_img'],
                    is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])


# train_dataset, test_dataset, _ = utils.create_dataset(
#     buffer_size, batch_size, data_format, data_dir)

    with strategy.scope():
        model = ArcFaceModel(size=cfg['input_size'],
                    backbone_type=cfg['backbone_type'],
                    num_classes=cfg['num_classes'],
                    head_type=cfg['head_type'],
                    embd_shape=cfg['embd_shape'],
                    w_decay=cfg['w_decay'],
                    margin=float(cfg['margin']),
                    logist_scale=float(cfg['logist_scale']), 
                    training=True,
                    use_pretrain=True)
        
        

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=cfg["base_lr"], momentum=0.9, nesterov=True)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        if cfg['checkpoint_dir']:
            checkpoint.restore(tf.train.latest_checkpoint(cfg["checkpoint_dir"]))
            print("Load checkpoint  from {}", tf.train.latest_checkpoint(cfg["checkpoint_dir"]))
    epochs = cfg["epochs"]
    enable_function = True 
    batch_size = cfg['batch_size']
    learning_rate = cfg['base_lr']
    trainer = Train(optimizer, checkpoint, cfg, epochs, enable_function, model, batch_size, strategy, learning_rate=learning_rate, step_per_epoch=steps_per_epoch)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    print('Training...')
    train_mode = "custom_loop"
    if train_mode == 'custom_loop':
        return trainer.custom_loop(train_dist_dataset,
        #  test_dist_dataset,
        strategy)
    elif train_mode == 'keras_fit':
        raise ValueError(
                '`tf.distribute.Strategy` does not support subclassed models yet.')
    else:
        raise ValueError(
            'Please enter either "keras_fit" or "custom_loop" as the argument.')

if __name__ == '__main__':
    app.run(main)

