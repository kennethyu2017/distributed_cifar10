# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass

  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
    tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
    tf.strided_slice(record_bytes, [label_bytes],
                     [label_bytes + image_bytes]),
    [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result.uint8image, result.label


# def _generate_image_and_label_batch(image, label, min_queue_examples,
#                                     batch_size, shuffle):
#   """Construct a queued batch of images and labels.
#
#   Args:
#     image: 3-D Tensor of [height, width, 3] of type.float32.
#     label: 1-D Tensor of type.int32
#     min_queue_examples: int32, minimum number of samples to retain
#       in the queue that provides of batches of examples.
#     batch_size: Number of images per batch.
#     shuffle: boolean indicating whether to use a shuffling queue.
#
#   Returns:
#     images: Images. 4D tensor of [batch_size, height, width, 3] size.
#     labels: Labels. 1D tensor of [batch_size] size.
#   """
#   # Create a queue that shuffles the examples, and then
#   # read 'batch_size' images + labels from the example queue.
#   num_preprocess_threads = 16
#   if shuffle:
#     images, label_batch = tf.train.shuffle_batch(
#         [image, label],
#         batch_size=batch_size,
#         num_threads=num_preprocess_threads,
#         capacity=min_queue_examples + 3 * batch_size,
#         min_after_dequeue=min_queue_examples)
#   else:
#     images, label_batch = tf.train.batch(
#         [image, label],
#         batch_size=batch_size,
#         num_threads=num_preprocess_threads,
#         capacity=min_queue_examples + 3 * batch_size)
#
#   # Display the training images in the visualizer.
#   tf.summary.image('images', images)
#
#   return images, tf.reshape(label_batch, [batch_size])
#

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      distorted_image = tf.image.random_brightness(image,
                                                   max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)
    else:
      distorted_image = tf.image.random_contrast(image,
                                                 lower=0.2, upper=1.8)
      distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)

    # The random_* ops do not necessarily clamp.
    # image = tf.clip_by_value(image, 0.0, 1.0)
    return distorted_image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  # with tf.name_scope(values=[image, height, width], name=scope,
  #                    default_name='eval_image'):
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  # image = tf.image.central_crop(image, central_fraction=0.875)
  #
  # # Resize the image to the original height and width.
  # image = tf.expand_dims(image, 0)
  # image = tf.image.resize_bilinear(image, [height, width],
  #                                  align_corners=False)
  # image = tf.squeeze(image, [0])
  return image


def distort_image(image, height, width, thread_id):
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
  distorted_image = distort_color(distorted_image, thread_id)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)

  return float_image


def image_preprocessing(image, train, thread_id=0):
  """Decode and preprocess one image for evaluation or training.

  Args:
    image:  Tensor
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  if train:
    image = distort_image(image, height, width, thread_id)
  else:
    image = eval_image(image, height, width)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def batch_inputs(data_dir, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    data_dir:
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].

  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = [os.path.join(data_dir, 'cifar-10-batches-bin/data_batch_%d.bin' % i)
                  for i in range(1, 6)]
    for f in data_files:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # Create filename_queue. will add a queue runner to enqueue_many filenames.
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
      examples_queue = tf.RandomShuffleQueue(
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.uint8, tf.int32])  # each element is [uint8img, label]
    else:
      examples_queue = tf.FIFOQueue(
        capacity=examples_per_shard + 3 * batch_size,
        dtypes=[tf.uint8, tf.int32])  # each element is [uint8img, label]

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        # Read examples from files in the filename queue.
        uint8img, label = read_cifar10(filename_queue)
        # reshaped_image = tf.cast(read_input.uint8image, tf.f loat32)
        enqueue_ops.append(examples_queue.enqueue([uint8img, label]))

      # each thread corresponds to one reader
      tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example = examples_queue.dequeue()
    else:  # single reader
      example = read_cifar10(filename_queue)

    image = tf.cast(example[0], tf.float32)
    label_index = example[1]
    label_index.set_shape([1])  # ensure shape

    images_and_labels = []
    # parallel pre-process in num_preprocess_threads
    for thread_id in range(num_preprocess_threads):
      image = image_preprocessing(image, train, thread_id)
      images_and_labels.append([image, label_index])

    image_batch, label_index_batch = tf.train.batch_join(
      images_and_labels,
      batch_size=batch_size,
      capacity=2 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    image_batch = tf.cast(image_batch, tf.float32)
    image_batch = tf.reshape(image_batch, shape=[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_index_batch = tf.reshape(label_index_batch, shape=[batch_size])

    # Display the training images in the visualizer.
    # tf.summary.image('images', image_batch)

    return image_batch, label_index_batch


def distorted_inputs(data_dir, batch_size=None, num_preprocess_threads=None,num_readers=None):
  """Generate batches of distorted versions of ImageNet images.

  Use this function as the inputs for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):  # tf.device can merge with /job:worker/task: #
    image_batch, label_batch = batch_inputs(
      data_dir, batch_size, train=True,
      num_preprocess_threads=num_preprocess_threads,
      num_readers=num_readers)
  return image_batch, label_batch


# def distorted_inputs(data_dir, batch_size):
#   """Construct distorted input for CIFAR training using the Reader ops.
#
#   Args:
#     data_dir: Path to the CIFAR-10 data directory.
#     batch_size: Number of images per batch.
#
#   Returns:
#     images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#     labels: Labels. 1D tensor of [batch_size] size.
#   """
#   filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
#                for i in xrange(1, 6)]
#   for f in filenames:
#     if not tf.gfile.Exists(f):
#       raise ValueError('Failed to find file: ' + f)
#
#   # Create a queue that produces the filenames to read.
#   filename_queue = tf.train.string_input_producer(filenames)
#
#   # Read examples from files in the filename queue.
#   read_input = read_cifar10(filename_queue)
#   reshaped_image = tf.cast(read_input.uint8image, tf.float32)
#
#   height = IMAGE_SIZE
#   width = IMAGE_SIZE
#
#   # Image processing for training the network. Note the many random
#   # distortions applied to the image.
#
#   # Randomly crop a [height, width] section of the image.
#   distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
#
#   # Randomly flip the image horizontally.
#   distorted_image = tf.image.random_flip_left_right(distorted_image)
#
#   # Because these operations are not commutative, consider randomizing
#   # the order their operation.
#   # NOTE: since per_image_standardization zeros the mean and makes
#   # the stddev unit, this likely has no effect see tensorflow#1458.
#   distorted_image = tf.image.random_brightness(distorted_image,
#                                                max_delta=63)
#   distorted_image = tf.image.random_contrast(distorted_image,
#                                              lower=0.2, upper=1.8)
#
#   # Subtract off the mean and divide by the variance of the pixels.
#   float_image = tf.image.per_image_standardization(distorted_image)
#
#   # Set the shapes of tensors.
#   float_image.set_shape([height, width, 3])
#   read_input.label.set_shape([1])
#
#   # Ensure that the random shuffling has good mixing properties.
#   min_fraction_of_examples_in_queue = 0.4
#   min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
#                            min_fraction_of_examples_in_queue)
#   print ('Filling queue with %d CIFAR images before starting to train. '
#          'This will take a few minutes.' % min_queue_examples)
#
#   # Generate a batch of images and labels by building up a queue of examples.
#   return _generate_image_and_label_batch(float_image, read_input.label,
#                                          min_queue_examples, batch_size,
#                                          shuffle=True)

#
# def inputs(eval_data, data_dir, batch_size):
#   """Construct input for CIFAR evaluation using the Reader ops.
#
#   Args:
#     eval_data: bool, indicating if one should use the train or eval data set.
#     data_dir: Path to the CIFAR-10 data directory.
#     batch_size: Number of images per batch.
#
#   Returns:
#     images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#     labels: Labels. 1D tensor of [batch_size] size.
#   """
#   if not eval_data:
#     filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
#                  for i in range(1, 6)]
#     num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#   else:
#     filenames = [os.path.join(data_dir, 'test_batch.bin')]
#     num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
#
#   for f in filenames:
#     if not tf.gfile.Exists(f):
#       raise ValueError('Failed to find file: ' + f)
#
#   # Create a queue that produces the filenames to read.
#   filename_queue = tf.train.string_input_producer(filenames)
#
#   # Read examples from files in the filename queue.
#   read_input = read_cifar10(filename_queue)
#   reshaped_image = tf.cast(read_input.uint8image, tf.float32)
#
#   height = IMAGE_SIZE
#   width = IMAGE_SIZE
#
#   # Image processing for evaluation.
#   # Crop the central [height, width] of the image.
#   resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
#                                                          height, width)
#
#   # Subtract off the mean and divide by the variance of the pixels.
#   float_image = tf.image.per_image_standardization(resized_image)
#
#   # Set the shapes of tensors.
#   float_image.set_shape([height, width, 3])
#   read_input.label.set_shape([1])
#
#   # Ensure that the random shuffling has good mixing properties.
#   min_fraction_of_examples_in_queue = 0.4
#   min_queue_examples = int(num_examples_per_epoch *
#                            min_fraction_of_examples_in_queue)
#
#   # Generate a batch of images and labels by building up a queue of examples.
#   return _generate_image_and_label_batch(float_image, read_input.label,
#                                          min_queue_examples, batch_size,
#                                          shuffle=False)
