
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import tensorflow.python.platform

import numpy
from scipy.misc import imresize, imsave
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import mrc

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64


tf.app.flags.DEFINE_string("train_root", '_autopick_train.star', "File suffix for star files with particle training data")
tf.app.flags.DEFINE_string("train_output", 'model.ckpt', "File to save training data")
tf.app.flags.DEFINE_integer("boxsize", 200, "Boxsize to use when extracting particles")
tf.app.flags.DEFINE_float("sigma_contrast", 0.0, "Sigma contrast to apply to image before particle extraction")
tf.app.flags.DEFINE_float("apix", 1.0, "Angstroms per pixel in mrc file")
tf.app.flags.DEFINE_float("lowpass", 0.0, "Lowpass filter resolution to apply")
tf.app.flags.DEFINE_float("highpass", 0.0, "Highpass filter resolution to apply")
tf.app.flags.DEFINE_float("gaussian_sigma", 0.0, "Gaussian filter sigma to apply")
tf.app.flags.DEFINE_integer("num_cores", 4, "Number of cores to use")
tf.app.flags.DEFINE_integer("num_epochs", 15, "Number of epochs for training")
tf.app.flags.DEFINE_integer("resized_box", 32, "Resize boxsize used for training")
FLAGS = tf.app.flags.FLAGS


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
      predictions.shape[0])

def particle_percent(predictions):
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1)) / predictions.shape[0])

def count_particle_totals(train_suffix, training_data = 0):
    particle_count = 0
    non_particle_count = 0
    allFiles = [ f for f in os.listdir(os.getcwd()) if f.endswith(train_suffix) ]
    for f in allFiles:
        with open(f, 'r') as input:
            for l in range(0,9):
                input.readline()
            for line in input:
                if line.strip():
                    if training_data:
                        fields = line.split()
                        if fields[5] == 'P':
                            particle_count += 1
                        else:
                            non_particle_count += 1
                    else:
                        particle_count += 1
    if training_data:
        return particle_count, non_particle_count
    else:
        return particle_count

def count_single_image_particle_totals(filename, training_data = 0):
    particle_count = 0
    non_particle_count = 0
    with open(filename, 'r') as input:
        for l in range(0,9):
            input.readline()
        for line in input:
            if line.strip():
                if training_data:
                    fields = line.split()
                    if fields[5] == 'P':
                        particle_count += 1
                    else:
                        non_particle_count += 1
                else:
                    particle_count += 1
    if training_data:
        return particle_count, non_particle_count
    else:
        return particle_count

def load_data(file_suffix, boxsize, image_size, sigma_contrast, num_images, apix, lowpass_filter, highpass_filter, gaussian_filter,  with_labels=0):
    image_array = 0
    if with_labels:
        labels_array = numpy.ndarray(shape=num_images*4, dtype=numpy.uint8)
    image_num = 0
    scale_factor = float(image_size / boxsize)
    allFiles = [ f for f in os.listdir(os.getcwd()) if f.endswith(file_suffix) ]
    for f in allFiles:
#        if image_num < 32:
#            print('Processing %s'%f)
        MRC_file_name = f[0:f.find(file_suffix)] + '.mrc'
        mrc_image = mrc.mrc()
        mrc_image.readFromFile(MRC_file_name)
        if lowpass_filter > 0:
            mrc_image.lowpass_filter(apix, lowpass_filter)
        if highpass_filter > 0:
            mrc_image.highpass_filter(apix, highpass_filter)
        if gaussian_filter > 0:
            mrc_image.apply_gaussian(gaussian_filter)
        mrc_image.getImageContrast(sigma_contrast)
        current_mrc_image_array = 0
        with open(f, 'r') as star:
            for l in range(0,9):
                star.readline()
            for line in star:
                if line.strip():
                    fields = line.split()
                    xc = int(float(fields[0]))
                    yc = int(float(fields[1]))
                    newData = mrc_image.generateScaled2DBox(xc, yc, boxsize)
#                    print('Min: %.1f Max: %.1f Avg: %.1f'% (newData.min(), newData.max(), newData.mean()))
                    newData = imresize(newData, scale_factor, mode='F')
                    if type(current_mrc_image_array) is int:
                        current_mrc_image_array = newData
                    else:
                        current_mrc_image_array = numpy.append(current_mrc_image_array, newData)
                    if with_labels:
                        rotated = numpy.rot90(newData)
                        current_mrc_image_array = numpy.append(current_mrc_image_array, rotated)
                        rotated = numpy.rot90(rotated)
                        current_mrc_image_array = numpy.append(current_mrc_image_array, rotated)
                        rotated = numpy.rot90(rotated)
                        current_mrc_image_array = numpy.append(current_mrc_image_array, rotated)
                    if with_labels:
                        if fields[5] == 'P':
                            labels_array[image_num] = 0
                            labels_array[image_num+1] = 0
                            labels_array[image_num+2] = 0
                            labels_array[image_num+3] = 0
                        else:
                            labels_array[image_num] = 1
                            labels_array[image_num+1] = 1
                            labels_array[image_num+2] = 1
                            labels_array[image_num+3] = 1
                    image_num += 4
        if type(image_array) is int:
            image_array = current_mrc_image_array
        else:
            if type(current_mrc_image_array) is not int:
                image_array = numpy.append(image_array, current_mrc_image_array)

    if with_labels:
        image_array = image_array.reshape(num_images*4, image_size, image_size, 1)
    else:
        image_array = image_array.reshape(num_images, image_size, image_size, 1)
    if with_labels:
        labels_array_hot = (numpy.arange(NUM_LABELS) == labels_array[:, None]).astype(numpy.float32)
        return image_array, labels_array_hot
    else:
        return image_array

def load_single_image_data(file_root, file_suffix, boxsize, image_size, sigma_contrast, apix, lowpass_filter, highpass_filter, gaussian_filter):
    image_array = 0
    image_num = 0
    scale_factor = float(image_size / boxsize)
    f = file_root + file_suffix
    MRC_file_name = file_root + '.mrc'
    mrc_image = mrc.mrc()
    mrc_image.readFromFile(MRC_file_name)
    if lowpass_filter > 0:
        mrc_image.lowpass_filter(apix, lowpass_filter)
    if highpass_filter > 0:
        mrc_image.highpass_filter(apix, highpass_filter)
    if gaussian_filter > 0:
        mrc_image.apply_gaussian(gaussian_filter)
    mrc_image.getImageContrast(sigma_contrast)
    with open(f, 'r') as star:
        for l in range(0,9):
            star.readline()
        for line in star:
            if line.strip():
                fields = line.split()
                xc = int(float(fields[0]))
                yc = int(float(fields[1]))
                newData = mrc_image.generateScaled2DBox(xc, yc, boxsize)
                newData = imresize(newData, scale_factor, mode='F')
                if type(image_array) is int:
                    image_array = newData
                else:
                    image_array = numpy.append(image_array, newData)
                image_num += 1

    image_array = image_array.reshape(image_num, image_size, image_size, 1)
    return image_array


def save_data(file_suffix, out_file_suffix, predictions):
    allFiles = [ f for f in os.listdir(os.getcwd()) if f.endswith(file_suffix) ]
    image_num = 0
    for f in allFiles:
        output_string = ''
        out_file_name = f[0:f.find(file_suffix)] + out_file_suffix
        with open(f, 'r') as star:
            for l in range(0,9):
                output_string += star.readline()
            for line in star:
                if line.strip():
                    if predictions[image_num][0]:
                        output_string += line
                    image_num += 1
        with open(out_file_name, 'w') as out:
            out.write(output_string)

def save_single_image_data(file_root, file_suffix, out_file_suffix, predictions):
    image_num = 0
    f = file_root + file_suffix
    output_string = ''
    out_file_name = file_root + out_file_suffix
    particles = 0
    non_particles = 0
    with open(f, 'r') as star:
        for l in range(0,9):
            output_string += star.readline()
        for line in star:
            if line.strip():
                if predictions[image_num][0] > predictions[image_num][1]:
                    output_string += line
                    particles += 1
                else:
                    non_particles += 1
                image_num += 1
    with open(out_file_name, 'w') as out:
        out.write(output_string)
    print('Saved ' + out_file_name + ' with ' + str(particles) + '/' + str(non_particles + particles) + '(%.1f%%)'%(100.0*particles / (particles + non_particles)))

def save_images(train_data, train_labels, image_size, num_images):
    for i in xrange(num_images):
        data = train_data[i]
        data = data.reshape((image_size, image_size))
        print('Image %d, label: '%i + str(train_labels[i]))
        if train_labels[i][0]:
            imsave('image_%3d_P.png'%i, data)
        else:
            imsave('image_%3d_N.png'%i, data)

def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
#    image_size = FLAGS.boxsize
  # image_size = IMAGE_SIZE
  image_size = FLAGS.resized_box

  particle_count, non_particle_count = count_particle_totals(FLAGS.train_root, training_data=1)
  print('Training Particles: ' + str(particle_count) + ' Non-particles: ' + str(non_particle_count) + ' in training set')
    
  train_data, train_labels = load_data(FLAGS.train_root, FLAGS.boxsize, image_size, FLAGS.sigma_contrast, particle_count + non_particle_count, FLAGS.apix, FLAGS.lowpass, FLAGS.highpass, FLAGS.gaussian_sigma, with_labels = 1)
    
  # save_images(train_data, train_labels, image_size, 32)

  validation_size = int(4 * (particle_count + non_particle_count) * 0.1)
    
#    print(train_labels)


  # Generate a validation set.
  validation_data = train_data[:validation_size, :, :, :]
  validation_labels = train_labels[:validation_size]
  train_data = train_data[validation_size:, :, :, :]
  train_labels = train_labels[validation_size:]
  num_epochs = FLAGS.num_epochs
  print('Loaded training data, using ' + str(validation_size) + ' particles for validation - %.1f%% particles' % particle_percent(validation_labels))

  train_size = train_labels.shape[0]

#  print(train_labels)
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, image_size, image_size, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, NUM_LABELS))
  # For the validation and test data, we'll just hold the entire dataset in
  # one constant node.
  validation_data_node = tf.constant(validation_data)

#  test_data_node = tf.constant(test_data)

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([32]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [image_size // 4 * image_size // 4 * 64, 512],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
  fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the minibatch, validation set and test set.
  train_prediction = tf.nn.softmax(logits)
#  test_prediction = tf.nn.softmax(model(test_data_node))

  # We'll compute them only once in a while by calling their {eval()} method.
  validation_prediction = tf.nn.softmax(model(validation_data_node))

  init_op = tf.initialize_all_variables()

  saver = tf.train.Saver()

  # Create a local session to run this computation.
  with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=FLAGS.num_cores,
                                      intra_op_parallelism_threads=FLAGS.num_cores)) as s:
    # Run all the initializers to prepare the trainable parameters.
    s.run(init_op)

    print('Initialized!')
    # Loop through training steps.
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % 100 == 0:
        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
        print('Step %d' % step)
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        validation_result = validation_prediction.eval()
        print('Validation error: %.1f%%' %
              error_rate(validation_result, validation_labels))
        save_path = saver.save(s, FLAGS.train_output, global_step=step)
        print('Step %d model saved in file %s' % (step, save_path))    
#        print('Validation result: ' + str(validation_result))
        sys.stdout.flush()
    # Save the picked particles...
    validation_result = validation_prediction.eval()
    print('Training completed, final validation error: %.1f%%' %error_rate(validation_result, validation_labels))
    save_path = saver.save(s, FLAGS.train_output)
    print('Trained model saved in file %s' % save_path)


if __name__ == '__main__':
  tf.app.run()
