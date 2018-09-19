from bbrbm import BBRBM
from tensorflow.examples.tutorials.mnist import input_data
from util import show_digit, plot_errors


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

bernouli_bernouli_rbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bernouli_bernouli_rbm.fit(mnist_images, n_epoches=1, batch_size=10)
bernouli_bernouli_rbm.save_weights("./weights/rbm_weights", "first")
plot_errors(errs)



#####
#Reconstruct the image by loading weights or just running the whole fit again
#####
bernouli_bernouli_rbm.load_weights("./weights/rbm_weights", "first")
IMAGE = 1


image = mnist_images[IMAGE]
image_rec = bernouli_bernouli_rbm.reconstruct(image.reshape(1,-1))

show_digit(image, "outputs/original.png")
show_digit(image_rec, "outputs/bb_reconstructed.png")
