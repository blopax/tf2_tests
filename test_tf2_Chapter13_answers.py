# ABOUT TF
#
# What is Tensorflow ?
#
# Around what is build tensorflow ? What are tensors ?
# On what rely tf computations ?
# How to use GPU with tf2 ?
"""
TF is a multiplatform/language library for efficient ML and Deep Learning developped by Google and open source

Built around directed graphs. Tensor are links to the nodes and describe transformation between input and output.

Computation can be made on GPU or TPU for much more efficiency. For GPU import tensorflow-gpu (but needs to have right hardware (linux + nvidia)).
"""


# BASIC OP ON TENSORS
# How to create tensors ?
# How to get basic properties ?
# What are some basic operations on tensors (4)?

# Instantiate 2 random tensors, one with uniform the other with normal distrib

# How to compute mean, sum, sd on a tensor? matrix product ? Lp norm ?

# How to split, stack, and concatenate ? (difference stack / concatenate ?)

# Completer stack: what constraints on tensors that will be stack?
# Packs the list of tensors in values into a tensor with rank one higher than each tensor in values, by packing them along the axis dimension.
# Given a list of length N of tensors of shape (A, B, C); if axis == 0 then the output tensor will have the shape ???. if axis == 1 then the output tensor will have the shape ???. Etc.

# Completer concat
# Concatenates the list of tensors values along dimension axis. If values[i].shape = [D0, D1, ... Daxis(i), ...Dn], the concatenated result has shape ???
# Constraints on all tensors ?


import tensorflow as tf
import numpy as np

t1 = tf.convert_to_tensor(np.random.rand(4, 3))
t2 = tf.convert_to_tensor(np.random.rand(3, 4))
t3 = tf.convert_to_tensor([1, 2, 3])

tf.cast(t1, 'int64')
tf.transpose(t1)
tf.reshape(t1, [2, 6])

tf.random.set_seed(1)
t4 = tf.random.uniform(shape=[3, 4], minval=0, maxval=1, name='uniform')
t5 = tf.random.normal(shape=[4, 3], mean=0.5, stddev=0.25, name='normal')

tf.linalg.matmul(t4, tf.transpose(t5), transpose_A=True)
tf.linalg.matmul(t4, tf.transpose(t5), transpose_a=True)

tf.math.reduce_mean(t5)
tf.math.reduce_std(t5)
tf.math.reduce_sum(t5)

tf.linalg.norm(t5, 2)

# How to split, stack, and concatenate ? (difference stack / concatenate ?)
tf.split(t3, num_or_size_splits=3, axis=0)
tf.split(t5, num_or_size_splits=[1, 2], axis=1)

tf.stack([t4, tf.transpose(t5)], axis=1)
tf.concatenate([t3, t4], axis=0)


# TF DATASETS API
# When can Keras api .fit() be used or not
# how to construct a tf Dataset from existing tensors, list or array ?
# What are some preprocessing Dataset:  how to get different rows of dataset ? how to create batches ?
# How to create dataset from t_x(features) and t_y(labels)
# How to apply on each element of dataset a transformation ? (for instance function(x,y) -> (x_normalized, y)
# When to shuffle ? batch ? how to go through different epochs ? --> how to iterate 3 times on a dataset that is shuffled in batch of 2 ?

# How to create a dataset from file on my locale storage disk ?
# Take the images folder to create a data set and display images

# show built in datasets. Import mnist et montrer caracteristiques
# creer train, test datasets et visualiser 10 images


x = tf.random.uniform((6,3), 3)
y = tf.random.normal((6,1), 1, 0.25)
t = tf.convert_to_tensor([1,2,3])
ds_t = tf.data.Dataset.from_tensor_slices(t)
ds_x = tf.data.Dataset.from_tensor_slices(x)
ds_y = tf.data.Dataset.from_tensor_slices(y)
ds_xy = tf.data.Dataset.zip(ds_x, ds_y)

for item in ds_t.shuffle(6).batch(3).repeat(2):
    print(item)

for item in ds_xy:
    print(item)

t3 = tf.random.uniform([4, 3], -4, 4)
t4 = tf.reshape(tf.convert_to_tensor([1, 0, 0, 1]), [4, 1])
ds2 = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(t3), tf.data.Dataset.from_tensor_slices(t4)))


import pathlib
images_dir = pathlib.Path('small_ds_10_images')
images_list = [str(item) for item in images_dir.iterdir()]
labels = [0] * len(images_list)
image_path_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(images_list))
image_ds = image_path_ds.map(lambda x: tf.image.decode_image(tf.io.read_file(x)))
feat_ds = tf.data.Dataset.zip((image_path_ds, image_ds))
label_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(labels))
total_ds = tf.data.Dataset.zip((feat_ds, label_ds))

import matplotlib.pyplot as plt

fig = plt.figure()

for i, item in enumerate(total_ds):
    ax = fig.add_subplot(5, 2, i + 1)
    image = item[0][1]
    ax.imshow(image)
    ax.set_title('Path {} \nLabel {}'.format(item[0][0], item[1]))

plt.show()


# show built in datasets. Import mnist et montrer caracteristiques
# creer train, test datasets et visualiser 10 images


import tensorflow_datasets as tfds
mnist_ds = tfds.builder('mnist')
mnist_ds.info
mnist_ds.download_and_prepare()
ds = mnist_ds.as_dataset(shuffle_files=False)

ds_train = (ds['train']).map(lambda item: (item['image'], item['label']))

ds_train = ds_train.batch(10)

batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure()
for i, (image, label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(image[:, :, 0])

plt.show()






# TF KERAS API
# Qu est ce que KEras ? Pourquoi l utiliser ?

# Comment creer une stack de layer ? differentes facons d empiler ?
# Qu est ce que le subclassing ?

# creer une linear regression avec subclassing sur donnes aleatoires simples. montrer les 10 points et courbe.
# utiliser iris dataset et creer un modele MLP simple (input layer = 4, hidden layer = 16 activation sigmoid, output = 3 activation softmax)
# avec keras sans subclassing + plot learning curves (loss and accuracy) + evaluate on test_dataset
# whn to use build or not ?

# how to save and load model. what format can a model be saved


# ACTIVATION FUNCTIONS
# how to choose ?
# What are the different examples ? draw on paper ?