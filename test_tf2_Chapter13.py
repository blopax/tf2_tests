#
#
# ABOUT TF
#
# What is Tensorflow ?
#
# Around what is build tensorflow ? What are tensors ?
# On what rely tf computations ?
# How to use GPU with tf2 ?



# BASIC OP ON TENSORS
# How to create tensors ?
# How to get basic properties ?
# What are some basic operations on tensors (4)?
#
# Instantiate 2 random tensors, one with uniform the other with normal distrib
#
# How to compute mean, sum, sd on a tensor? matrix product ? nLp norm ?
#
# How to split, stack, and concatenate ? (difference stack / concatenate ?)


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