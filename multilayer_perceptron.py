#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:05:53 2017

@author: 390771
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Builds the network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""
import tensorflow as tf

# The  dataset has 2 classes
NUM_CLASSES = 2

MFCC_PIXELS = 13
# 建网络模型
def inference(features, hidden1_units, hidden2_units):

    """Build the  model up to where it may be used for inference.

    Args:
      MFCCfeatures: features placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # 设置网络层数
    # Hidden 1
    with tf.name_scope('hidden1'):
      weights = tf.Variable(
        tf.truncated_normal([MFCC_PIXELS, hidden1_units]),name='weights')
      biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
      hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
      #Add a scalar summary for the snapshot weights
      tf.summary.histogram('weigtht', weights)
      tf.summary.histogram('hidden1/biases', biases)
    
    # Hidden 2
    with tf.name_scope('hidden2'):
      weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units]),name='weights')
      biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
      #Add a scalar summary for the snapshot weights
      tf.summary.histogram('hidden2/weigtht', weights)
      tf.summary.histogram('hidden2/biases', biases)
    # 输出网络层
    # Linear
    with tf.name_scope('softmax_linear'):
      weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES]),name='weights')
      biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
      logits = tf.matmul(hidden2, weights) + biases
      tf.summary.histogram('softmax_linear/weigtht', weights)
      tf.summary.histogram('softmax_linear/biases', biases)
    return logits

def loss(logits, labels):

    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    # 损失函数 交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  
    return tf.reduce_mean(cross_entropy)


def training(loss, learning_rate):

    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    # 画图使用的
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    # 设置梯度下降学习
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #Add a scalar summary for the snapshot optimizer
    # Create a variable to track the global step.
    # 查看全局变量
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    # 根据目标损失函数函数最小
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # 计算准确度
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy
