import numpy as np
from bbrbm import BBRBM
from gbrbm import GBRBM
from tensorflow.examples.tutorials.mnist import input_data
from util import show_digit, plot_errors

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

gaussian_bernouli_rbm = GBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, err_function='mse', use_tqdm=True, sample_visible=False, sigma=0.01)
errors = gaussian_bernouli_rbm.fit(mnist_images, n_epoches=1, batch_size=10)
gaussian_bernouli_rbm.save_weights("./weights/grbm_weights", "first")

plot_errors(errors)




######
#Reconstruct the image by loading weights or just running the whole fit again
######
gaussian_bernouli_rbm.load_weights("./weights/grbm_weights", "first")
IMAGE = 1



image = mnist_images[IMAGE]
image_rec = gaussian_bernouli_rbm.reconstruct(image.reshape(1,-1))

show_digit(image, "outputs/original.png")
show_digit(image_rec, "outputs/gb_reconstructed4.png")