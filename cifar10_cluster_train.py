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

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
from cifar10_model import inference,loss, add_loss_summaries,MOVING_AVERAGE_DECAY,\
  collect_batchnorm_updates,maybe_download_and_extract
from cifar10_dataset import Cifar10DataSet
from cifar10_input import distorted_inputs
from tensorflow.core.protobuf import saver_pb2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '192.168.1.102:8888,192.168.1.103:8888',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '192.168.1.100:8888',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_cluster_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_cluster_chkpnts',
#                            """Directory where to write chkpnts """
#                            """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_integer('image_size', 299,
#                             """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor',16,
                            """Input queue memory factor""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """If use float point 16""")

tf.app.flags.DEFINE_string('data_dir', './dataset',
                            """cifar10 data directory""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')



def replica_loss(images, labels,var_device_fn, is_chief):
  """Calculate the total loss on a single replica tower running the CIFAR model.

  Args:
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
     :param is_chief:
     :param var_device_fn:
  """

  # Build inference Graph.
  logits = inference(images,var_device_fn)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  cross_entropy_loss = loss(logits, labels)

  losses_list = [cross_entropy_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  total_loss = tf.add_n(losses_list, name='total_loss')

  if is_chief:  # summary loss and loss avgs on chief only.
    loss_avg_op = add_loss_summaries(losses_list, total_loss)
    # Add dependency to compute loss_averages.
    with tf.control_dependencies([loss_avg_op]):
      total_loss = tf.identity(total_loss)

  return total_loss

  # REG LOSS ????
  # Assemble all of the losses for the current tower only.
  # losses = tf.get_collection('losses', scope)

  # # Calculate the total loss for the current tower.
  # total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  # for l in losses + [total_loss]:
  #   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  #   # session. This helps the clarity of presentation on tensorboard.
  #   loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
  #   tf.summary.scalar(loss_name, l)

  # return total_loss

def moving_avg_vars(is_chief,global_step):
  if is_chief:
    # Track the moving averages of all trainable variables.
    # Note that we maintain a 'double-average' of the BatchNormalization
    # global statistics.
    # This is not needed when the number of replicas are small but important
    # for synchronous distributed training with tens of workers/replicas.
    exp_moving_averager = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)

    # include loss average slot var defined by chief,batch norm moving vars(shared
    # by all workers).
    # passed to syncReplicaOpt, but only sync_token_op will triger moving updates.
    # note: loss avg vars will be located on worker tasks.
    variables_to_average = (
      tf.trainable_variables() + tf.moving_average_variables())
    # Add histograms for model variables.
    for var in variables_to_average:
      tf.summary.histogram(var.op.name, var)

  else: #other workers
    exp_moving_averager = None
    variables_to_average = None

  return exp_moving_averager, variables_to_average

def _vars_on_job(job):
  """Returns all variables and `SaveableObject`s that must be checkpointed.

  Args:
    scope: (Optional.) A string. If supplied, the resulting list is filtered
      to include only items whose `name` attribute matches `scope` using
      `re.match`. Items without a `name` attribute are never returned if a
      scope is supplied. The choice of `re.match` means that a `scope` without
      special tokens filters by prefix.

  Returns:
    A list of `Variable` and `SaveableObject` to be checkpointed
  """
  # TODO(andreasst): make this function public once things are settled.

  ps_vars=[]
  vars= (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, None) +
          tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, None) +
         tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,None))
  for v in vars:
    if job in v.device:
      ps_vars.append(v)
  return ps_vars




#variable device chooser.
class VariableDeviceChooser(object):
  def __init__(self,ps_tasks, ps_device):
    #  to choose ps tasks round-robin(default strategy).
    self._task_choose_fn = tf.train.replica_device_setter(ps_tasks,
                                                 ps_device,worker_device=None)

  def device_function(self,op):
    return self._task_choose_fn(op) + '/CPU:0'


# def make_log_dir(logdir,delete=False):
#   if tf.gfile.Exists(logdir):
#     if delete:
#       tf.gfile.DeleteRecursively(logdir)
#     else:
#       tf.gfile.Rename(logdir, logdir + '_bak_{}'.format(time.time()))
#   return tf.gfile.MakeDirs(logdir)


def train(target, cluster_spec, server_def):
  # Number of workers and parameter servers are inferred from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief.
  is_chief = (FLAGS.task_id == 0)

  var_device_chooser = VariableDeviceChooser(ps_tasks=num_parameter_servers,
                                              ps_device="/job:ps")

  with tf.device(var_device_chooser.device_function): # on ps task
    global_step = tf.train.create_global_step()

  # Ops are assigned to worker by default.  # default on GPU:0. so one worker owns one gpu.
  with tf.device('/job:worker/task:%d' % FLAGS.task_id):
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (Cifar10DataSet.num_examples_per_epoch('train') /
                             FLAGS.batch_size)
    # Decay steps need to be divided by the number of replicas to aggregate.
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
                      num_replicas_to_aggregate)

    # Decay the learning rate exponentially based on the number of steps.
    # not create variable.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    # Add a summary to track the learning rate tensor.
    tf.summary.scalar('learning_rate', lr)

    # Adam
    adam_opt = tf.train.AdamOptimizer(lr)

    # Get images and labels for CIFAR-10. each worker has local dataset files
    # and create local queue for input to model.
    image_batch, label_batch = distorted_inputs(FLAGS.data_dir, FLAGS.batch_size,
                                                FLAGS.num_preprocess_threads,FLAGS.num_readers)

    # Calculate the loss for one replica of the CIFAR model.
    total_loss = replica_loss(image_batch, label_batch,
                        var_device_chooser.device_function,is_chief)

    # moving avg of all trainable variables and BN moving vars.
    # so we make chief to run apply op. for non-chief, avger/vars_to_avg are None.
    avger,vars_to_avg = moving_avg_vars(is_chief, global_step)

    # Create synchronous replica optimizer.
    sync_opt = tf.train.SyncReplicasOptimizer(
      adam_opt,
      replicas_to_aggregate=num_replicas_to_aggregate,
      total_num_replicas=num_workers,
      variable_averages=avger,  # only sync_token_op will triger moving updates.
      variables_to_average=vars_to_avg)

    # each worker will update bn moving mean and variance
    batchnorm_updates_op = collect_batchnorm_updates()
    if batchnorm_updates_op:
      with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(total_loss)

    # each worker Compute gradients
    grads = sync_opt.compute_gradients(total_loss)

    # Add histograms for gradients.
    # for non-chief, how can get summary without running summary_op?
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # each worker applys grads (accumulate grads actually)
    apply_grads_op = sync_opt.apply_gradients(grads, global_step=global_step)

    # each worker, including chief, will run train_op
    with tf.control_dependencies([apply_grads_op]):
      train_op = tf.identity(total_loss, name='train_op')

    # Get chief queue_runners and init_tokens, which is used to synchronize
    # replicas.
    # chief_queue_runners = [sync_opt.get_chief_queue_runner()]
    # init_tokens_op = sync_opt.get_init_tokens_op()

    # Create a saver.
    # NOTE : use sharded solution. we can :
    # 1. use SaverDef.V2 version, the merge_v2_op will merge all chk pnt files on different
    # ps tasks, so a distributed file system,e.g. HDFS,is necessary for saver_def.V2.
    # 2. use SaverDef.V1 without merge_op in case there is no distributed file system.
    # 3. either case, we need to ensure train_dir writable. normally use '/tmp/xxx_train'.
    saver = tf.train.Saver(var_list=_vars_on_job('job:ps'),sharded=True,write_version=saver_pb2.SaverDef.V1)

    # init local loss_avg vars which are not to be saved/restored. and local init op
    # will be runned after restore by Supervisor.
    local_init_op = tf.variables_initializer(var_list=_vars_on_job('job:{}/task:{}'.format(
      server_def.job_name, server_def.task_index)))

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    init_tokens_op = sync_opt.get_init_tokens_op()

    # collect all var initializers for chief to run in Supervisor.
    init_var_op = tf.global_variables_initializer()

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.train_dir,
                             init_op=init_var_op,
                             local_init_op=local_init_op,
                             summary_op=None, # dont start summary thread in supervisor.
                             global_step=global_step,
                             saver=saver,
                             save_model_secs=FLAGS.save_interval_secs)
    tf.logging.info('%s Supervisor' % datetime.now())

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)

    # get session:
    # for chief: prepare_session  for other workers: wait_for_session
    # we dont start standard service here, cause which will start queue runners including input queue runners.
    # NOTE: the prepare session will run restore_op of saver, but restore_op is co-located
    # with variables to be restored, on ps tasks. so when chief run restore_op, it will be
    # runned on ps task, but the check point files are saved on chief worker, so restore_op
    # can not find check point file.
    # TO resolve: 1. use sharded saver. 2.copy the check point files to ps tasks.  3.modify the generation of restore_op
    # in saver, not co-locate with variables.
    sess = sv.prepare_or_wait_for_session(target, config=sess_config,start_standard_services=False)

    # for chief to start standard service
    if is_chief:
      tf.logging.info('Started standard service.')
      # start flush summary,chkpnt service threads..
      sv.start_standard_services(sess)

    # for each worker to start input queue runners
    # Note: the sync_opt.chief_queue_runner was not added to GraphKeys.QUEUE_RUNNERS,
    # so for each worker, we can collect all queue runners except chief_queue_runner here.
    input_queue_runners=tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    input_threads = sv.start_queue_runners(sess,input_queue_runners)
    tf.logging.info('Started %d threads for processing input data.',
                    len(input_threads))

    # for chief: start take_grad -> sync token threads, and init token queue
    if is_chief:
      # the sync_opt.chief_queue_runner was not added to GraphKeys.QUEUE_RUNNERS.
      # we explicitly run chief_queue_runner.
      sv.start_queue_runners(sess, [sync_opt.get_chief_queue_runner()])
      sess.run(init_tokens_op)

    # ==== start training loop ====
    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    while not sv.should_stop():
      try:
        start_time = time.time()
        loss_value, step = sess.run([train_op, global_step])
        # check Nan.
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        if step > FLAGS.max_steps:
          break
        duration = time.time() - start_time

        if step % 30 == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          format_str = ('Worker %d: %s: step %d, loss = %.2f'
                        '(%.1f examples/sec; %.3f  sec/batch)')
          tf.logging.info(format_str %
                          (FLAGS.task_id, datetime.now(), step, loss_value,
                           examples_per_sec, duration))

        # Determine if the summary_op should be run on the chief worker.
        if is_chief and next_summary_time < time.time():
          tf.logging.info('Running Summary operation on the chief.')
          summary_str = sess.run(summary_op)
          sv.summary_computed(sess, summary_str)
          tf.logging.info('Finished running Summary operation.')

          # Determine the next time for running the summary.
          next_summary_time += FLAGS.save_summaries_secs
      except:
        if is_chief:
          tf.logging.info('Chief got exception while running!')
        raise

    # Stop the supervisor.  This also waits for service threads to finish.
    sv.stop()

    # save the last model
    if is_chief:
      saver.save(sess, save_path=os.path.join(FLAGS.checkpoint_dir, 'last_model.ckpt'),
                 global_step=global_step)


def main(argv=None):  # pylint: disable=unused-argument
  # worker download dataset
  if not FLAGS.job_name == 'ps':
    maybe_download_and_extract(FLAGS.data_dir)

  # define cluster
  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)

  cluster_spec = tf.train.ClusterSpec({
                  'ps':ps_hosts,
                  'worker':worker_hosts
                    })
  # start server
  server=tf.train.Server(cluster_spec,job_name=FLAGS.job_name,task_index=FLAGS.task_id,
                         protocol=FLAGS.protocol,start=True) #start gprc master/worker service threads

  # we build train_dir on both ps(for chk pnt shard) and worker(for summary)
  if not tf.gfile.Exists(FLAGS.train_dir):
   tf.gfile.MakeDirs(FLAGS.train_dir)

  if FLAGS.job_name == 'ps':
    server.join()  # block until server stops
  else:
    train(server.target,cluster_spec, server.server_def)


if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()


# if __name__ == '__main__':
#   chooser=VariableDeviceChooser(3,ps_device='/job:ps')
#   for _ in range(3):
#     var=tf.Variable(0)
#     device = chooser.device_function(var.op)
#     print('device :{}'.format(device))

