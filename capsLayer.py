import numpy as np
import tensorflow as tf

from config import cfg
from utils import reduce_sum
from utils import softmax


epsilon = 1e-9


class CapsLayer:
  """Capsule Layer
    :param num_outputs: the number of capsule in this layer
    :param vec_len: the length of the output vector of a capsule.
    :param with_routing: boolean, this capsule is routing with the lower-level
      capsule
    :param layer_type: string, one of 'FC' or 'CONV', the type of this layer,
      for the future expansion capability
  """
  def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
    self.num_outputs = num_outputs
    self.vec_len = vec_len
    self.with_routing = with_routing
    self.layer_type = layer_type

  def __call__(self, input_vector, kernel_size=None, stride=None):
    if self.layer_type == 'CONV':
      self.kernel_size = kernel_size
      self.stride = stride

      if not self.with_routing:
        # the PrimaryCaps layer, a convolutional layer
        # input: [batch_size, 20, 20, 256]
        assert input_vector.get_shape() == [cfg.batch_size, 20, 20, 256]
        """
        # version 1, computational expensive
        capsules = []
        for i in range(self.vec_len):
          # each capsule i: [batch_size, 6, 6, 32]
          with tf.variable_scope('ConvUnit_' + str(i)):
            caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                              self.kernel_size, self.stride,
                                              padding="VALID",
                                              activation_fn=None)
            caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
            capsules.append(caps_i)
        assert capsules[0].get_shape() == [cfg.batch_size, -1, 1, 1]
        capsules = tf.concat(capsules, axis=2)
        """

        # version 2, equivalent to version 1 but higher computational efficiency
        # NOTE: I can't find out any words from the paper whether the PrimaryCap
        # convolution does a ReLU activation or not before squashing function,
        # but experiment show that using ReLU get a higher test accuracy. So,
        # which one to use will be your choice
        capsules = tf.contrib.layers.conv2d(input_vector,
                                            self.num_outputs*self.vec_len,
                                            self.kernel_size,
                                            self.stride, padding="VALID",
                                            activation_fn=tf.nn.relu)
        # capsules = tf.contrib.layers.conv2d(input_vector,
        #                                     self.num_outputs*self.vec_len,
        #                                     self.kernel_size,
        #                                     self.stride, padding="VALID",
        #                                     activation_fn=None)
        capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

        # [batch_size, 1152, 8, 1]
        capsules = squash(capsules)
        assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
        return capsules

    if self.layer_type == "FC":
      if self.with_routing:
        # the DigitCaps layer, a fully connected layer
        # Reshape the input into [batch_size, 1152, 1, 8, 1]
        self.input = tf.reshape(input_vector,
                                shape=(cfg.batch_size, -1, 1,
                                       input_vector.shape[-2].value, 1))
        with tf.variable_scope('routing'):
          b_ij = tf.constant(np.zeros([cfg.batch_size,
                                       input_vector.shape[1].value,
                                       self.num_outputs,
                                       1,
                                       1],
                             dtype=np.float32))
          capsules = routing(self.input, b_ij)
          capsules = tf.squeeze(capsules, axis=1)

        return capsules


def routing(input, b_ij):
  """The routing algorithm
    :param input: A Tensor of shape `[batch, num_caps_l, 1,  len(u_i), 1]`,
      where `num_caps_l=1152` meaning the number  of capusle in layer l,
      and `length(u_i)=8`.
    :param b_ij: input weights from layer l to layer l+1
    :return: A Tensor of shape `[batch, num_caps_l_plus_1, len(v_j), 1]`
      representing the vector output 'v_j` in layer l+1
    Notes:
      `u_i` represents the vector output of capsule i in layer l, and `v_j`
      represents the vector output of capsule j in the layer l_1

  """
  # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1
  initializer = tf.random_normal_initializer(stddev=cfg.stddev)
  W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                      initializer=initializer)
  biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))

  # Eq.2, cal u_hat
  # Since tf.matmul is a time-consuming op, A better solution is using
  # element-wise multiply, reduce-sum and reshape ops instead. Matmul
  # [a, b] * [b, c] is equal to a series ops as element-wise multiply
  # [a*c, b] * [a*c, b], reduce_sum at axis=1 and reshape to [a, c]
  input = tf.tile(input, [1, 1, 160, 1, 1])
  assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

  u_hat = reduce_sum(W * input, axis=3, keepdims=True)
  u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])
  assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

  # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back
  # from u_hat_stop to u_hat
  u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
  v_j = None
  for r_iter in range(cfg.iter_routing):
    with tf.variable_scope('iter_' + str(iter)):
      c_ij = softmax(b_ij, axis=2)

      if r_iter == cfg.iter_routing - 1:
        s_j = tf.multiply(c_ij, u_hat)
        s_j = reduce_sum(s_j, axis=1, keepdims=True) + biases
        assert s_j.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
        v_j = squash(s_j)
        assert v_j.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
      elif r_iter < cfg.iter_routing - 1:
        s_j = tf.multiply(c_ij, u_hat_stopped)
        s_j = reduce_sum(s_j, axis=1, keepdims=True) + biases
        v_j = squash(s_j)

        v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])
        u_produce_v = reduce_sum(u_hat_stopped * v_j_tiled,
                                 axis=3, keepdims=True)
        assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]
        b_ij += u_produce_v
  assert v_j is not None
  return v_j


def squash(vector):
  """Squashing function corresponding to Eq. 1
     :param vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1]
            or [batch_size, num_caps, vec_len, 1].
     :returns: A tensor with the same shape but squashed in `vec_len` dimension.
  """
  vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
  scalar_factor = vec_squared_norm / (1 + vec_squared_norm)
  scalar_factor = scalar_factor / tf.sqrt(vec_squared_norm + epsilon)
  return scalar_factor * vector  # element-wise
