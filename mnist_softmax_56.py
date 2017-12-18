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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

FLAGS = None

sample_size = 5000
print("Softmax MNIST. 28X28. Sample size: " + str(sample_size))
pd_frame = pd.read_csv('train_tsne_56.csv', header=None)
pd_frame_test = pd.read_csv('test_tsne_56.csv', header=None)
cols = [0]
label_a_temp = pd_frame.as_matrix(pd_frame.columns[cols])
label_a_test = pd_frame_test.as_matrix(pd_frame_test.columns[cols])
pd_frame = pd_frame.drop(pd_frame.columns[cols], axis=1)
pd_frame_test = pd_frame_test.drop(pd_frame_test.columns[cols], axis = 1)
data_a_temp = pd_frame.as_matrix(columns = pd_frame.columns)
data_a_test = pd_frame_test.as_matrix(columns = pd_frame_test.columns)


idx = np.arange(0 , 50000)
np.random.shuffle(idx)
idx = idx[:sample_size]
data_shuffle = [data_a_temp[ i] for i in idx]
labels_shuffle = [label_a_temp[ i] for i in idx]


data_a = data_shuffle
label_a = labels_shuffle

def one_hot(i):
    a = np.zeros(10, 'uint8')
    a[i] = 1
    return a

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    for i,j in enumerate(labels_shuffle):
        labels_shuffle[i] = one_hot(j)
    batch = []
    batch.append(np.asfarray(data_shuffle))
    batch.append(np.asfarray(labels_shuffle))
    return batch
def next_batch_test(num, data, labels):
    idx = np.arange(0 , len(data))
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    for i,j in enumerate(labels_shuffle):
        labels_shuffle[i] = one_hot(j)
    batch = []
    batch.append(np.asfarray(data_shuffle))
    batch.append(np.asfarray(labels_shuffle))
    return batch


def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 3136])
  W = tf.Variable(tf.zeros([3136, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(20000):
    batch_xs, batch_ys = next_batch(50, data_a, label_a)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: batch_xs,
                                      y_: batch_ys}))

  # Test trained model
  batch_test = next_batch_test(10000, data_a_test, label_a_test)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: batch_test[0],
                                      y_: batch_test[1]}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
