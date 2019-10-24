import math
import cv2
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from squeeze_net import create_convnet

########################################################################
# function needed by the FCN class
def initialize_dataset_config(dataset_name=None, subset=None, fold=None):
  DATASET_NAME = 'gehler'
  FOLD = 0
  if dataset_name is not None:
    DATASET_NAME = dataset_name
    SUBSET = subset
    FOLD = int(fold)
  global TRAINING_FOLDS, TEST_FOLDS
  if DATASET_NAME == 'gehler':
    T = FOLD
    if T != -1:
      TRAINING_FOLDS = ['g%d' % (T), 'g%d' % ((T + 1) % 3)]
      TEST_FOLDS = ['g%d' % ((T + 2) % 3)]
    else:
      TRAINING_FOLDS = []
      TEST_FOLDS = ['g0', 'g1', 'g2']
  elif DATASET_NAME == 'cheng':
    subset = SUBSET
    T = FOLD
    TRAINING_FOLDS = ['c%s%d' % (subset, T), 'c%s%d' % (subset, (T + 1) % 3)]
    TEST_FOLDS = ['c%s%d' % (subset, (T + 2) % 3)]
  elif DATASET_NAME == 'multi':
    TEST_FOLDS = ['multi']
  return TRAINING_FOLDS, TEST_FOLDS
########################################################################

class FCN:

  def __init__(self, sess=None, name=None, kwargs={}, sq_path=None):
    global TRAINING_FOLDS, TEST_FOLDS
    self.name = name
    self.sq_path = sq_path
    self.wd = 5.7e-5
    TRAINING_FOLDS, TEST_FOLDS = initialize_dataset_config(**kwargs)
    self.training_data_provider = None
    self.sess = sess
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=tf.keras.regularizers.l2(5.7e-5)):
      self.build()
    tf.compat.v1.global_variables_initializer()
    # Store the test-time networks, for images of different resolution
    self.test_nets = {}
    self.saver = tf.compat.v1.train.Saver(max_to_keep=0)


  @staticmethod
  def build_branches(images, dropout, sq_path):
    images = tf.clip_by_value(images, 0.0, 65535.0)
    images = images * (1.0 / 65535)
    feed_to_fc = []

    with tf.compat.v1.variable_scope('AlexNet'):
        alex_images = (tf.pow(images, 1.0 / 2.2) *
                       255.0)[:, :, :, ::-1]
        alex_outputs = create_convnet(alex_images, sq_path)
    
    alexnet_output = alex_outputs['features_out']
    feed_to_fc.append(alexnet_output)
    feed_to_fc = tf.concat(axis=3, values=feed_to_fc)
    fc1 = slim.conv2d(
        feed_to_fc, 64, [6, 6], scope='fc1')
    fc1 = slim.dropout(fc1, dropout)
    fc2 = slim.conv2d(fc1, 3, [1, 1], scope='fc2', activation_fn=None)
    return fc2

  # Build the network
  def build(self):
    self.dropout = tf.compat.v1.placeholder(tf.float32, shape=(), name='dropout')
    # We don't use per_patch_weight any more.
    self.per_patch_weight = tf.compat.v1.placeholder(
        tf.float32, shape=(), name='pre_patch_weight')
    per_patch_weight = self.per_patch_weight
    self.learning_rate = tf.compat.v1.placeholder(
        tf.float32, shape=(), name='learning_rate')
    # ground truth, actually
    self.illums = tf.compat.v1.placeholder(tf.float32, shape=(None, 3), name='illums')
    # input images
    self.images = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, 512, 512, 3),
        name='images')

    with tf.compat.v1.variable_scope('FCN'):
      fc2 = self.build_branches(self.images, self.dropout, self.sq_path)

    self.per_pixel_est = fc2
    self.illum_normalized = tf.nn.l2_normalize(
        tf.reduce_sum(fc2, axis=(1, 2)), 1)
    self.global_loss = self.get_angular_loss(
        tf.reduce_sum(fc2, axis=(1, 2)), self.illums, 0.0)
    self.per_patch_loss = self.get_angular_loss(
        fc2, self.illums[:, None, None, :], 0.0)
    self.loss = (
        1 - per_patch_weight
    ) * self.global_loss + self.per_patch_weight * self.per_patch_loss
    scalar_summaries = []
    scalar_summaries.append(
        tf.compat.v1.summary.scalar('per_patch_loss', self.per_patch_loss))
    scalar_summaries.append(
        tf.compat.v1.summary.scalar('full_image_loss', self.global_loss))
    scalar_summaries.append(tf.compat.v1.summary.scalar('loss', self.loss))
    self.scalar_summaries = tf.compat.v1.summary.merge(scalar_summaries)
    conv_scopes = []
    conv_scopes.append('FCN/AlexNet/conv1')
    image_summaries = []
    self.merge_summaries = tf.compat.v1.summary.merge_all()
    reg_losses = tf.add_n(tf.compat.v1.losses.get_regularization_losses())
    self.total_loss = self.loss + reg_losses
    self.train_step_adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(
        self.total_loss)
    var_list1 = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/AlexNet')
    var_list2 = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/fc1') + tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='FCN/fc2')
    opt1 = tf.compat.v1.train.AdamOptimizer(self.learning_rate * 1e-1)
    opt2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
    grads = tf.gradients(self.total_loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    self.train_step_sgd = tf.group(train_op1, train_op2)


  def test_external(self, images, sq_path, scale=1.0, fns=None, show=False, write=True):
    illums = []
    confidence_maps = []
    for img, filename in zip(images, fns):
      if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
      shape = img.shape[:2]
      if shape not in self.test_nets:
        aspect_ratio = 1.0 * shape[1] / shape[0]
        if aspect_ratio < 1:
          target_shape = (400, 400 * aspect_ratio)
        else:
          target_shape = (400 / aspect_ratio, 400)
        target_shape = tuple(map(int, target_shape))
        test_net = {}
        test_net['illums'] = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 3), name='test_illums')
        test_net['images'] = tf.compat.v1.placeholder(
            tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
        with tf.compat.v1.variable_scope("FCN", reuse=True):
          test_net['pixels'] = FCN.build_branches(test_net['images'], 1.0, sq_path)
          test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
        
        self.test_nets[shape] = test_net

      test_net = self.test_nets[shape]
      pixels, est = self.sess.run(
          [test_net['pixels'], test_net['est']],
          feed_dict={
              test_net['images']: img[None, :, :, :],
              test_net['illums']: [[1, 1, 1]]
          })

      est = est[0]
      est /= np.linalg.norm(est)
      pixels = pixels[0]
      confidences = np.linalg.norm(pixels, axis=2)
      confidence_maps.append(confidences)
      ind = int(confidences.flatten().shape[0] * 0.95)
      illums.append(est)
    return illums, confidence_maps, est

    
  def load_absolute(self, fn):
    self.saver.restore(self.sess, fn)


  def get_angular_loss(self, vec1, vec2, length_regularization=0.0):
    with tf.name_scope('angular_error'):
      safe_v = 0.999999
      if len(vec1.get_shape()) == 2:
        illum_normalized = tf.nn.l2_normalize(vec1, 1)
        _illum_normalized = tf.nn.l2_normalize(vec2, 1)
        dot = tf.reduce_sum(illum_normalized * _illum_normalized, 1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        length_loss = tf.reduce_mean(
            tf.maximum(tf.math.log(tf.reduce_sum(vec1**2, axis=1) + 1e-7), 0))
      else:
        assert len(vec1.get_shape()) == 4
        illum_normalized = tf.nn.l2_normalize(vec1, 3)
        _illum_normalized = tf.nn.l2_normalize(vec2, 3)
        dot = tf.reduce_sum(illum_normalized * _illum_normalized, 3)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        length_loss = tf.reduce_mean(
            tf.maximum(tf.math.log(tf.reduce_sum(vec1**2, axis=3) + 1e-7), 0))
      angle = tf.acos(dot) * (180 / math.pi)

      return tf.reduce_mean(angle) + length_loss * length_regularization