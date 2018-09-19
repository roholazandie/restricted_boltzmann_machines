import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


def show_digit(x, filename):
    fig = plt.figure()
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.savefig(filename, dpi=fig.dpi)
    plt.show()


def plot_errors(x):
    plt.plot(x)
    plt.show()