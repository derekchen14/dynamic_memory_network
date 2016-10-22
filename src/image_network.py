import sys, os
import math
import random
import time
from glob import iglob

import numpy
import tensorflow as tf

BATCH_SIZE = 10
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_DEPTH = 3
CLASS_COUNT = 10
SAVE_INTERVAL = 100

INITIAL_LEARNING_RATE = 0.001 
LEARNING_RATE_DECAY = 0.999

#activation_function = tf.nn.relu
activation_function = tf.nn.elu
#activation_function = tf.tanh

def basic_image_iterator(path, batch_size):
	"""A generator which yields a 4-D numpy tensor of size batch,height,width,depth and a size of batch,labels label.."""
	labels = None
	examples = None
	with open(os.path.join(path, "train_y.bin"), "rb") as fin:
		labels = numpy.fromfile(fin, dtype=numpy.uint8)
	with open(os.path.join(path, "train_X.bin"), "rb") as fin:
		data = numpy.fromfile(fin, dtype=numpy.uint8)
		data = numpy.reshape(data, (-1, 3, 96, 96)) # Column major -> 3x96x96 tensor.
		data = numpy.transpose(data, (0, 2, 3, 1)) # Go from B D H W -> B H W D
		data = numpy.asarray(data / 255.0, dtype=numpy.float) # TODO: Divide by 255?
	while True:
		# Select random indices from the range.
		#sample_indices = numpy.random.randint(low=0, high=data.shape[0], size=(batch_size,))
		# DEBUG: This will always train on the same example to overfit.  
		sample_indices = numpy.zeros(shape=(batch_size,), dtype=numpy.uint8)
		x = data[sample_indices, :, :, :]
		y = numpy.zeros(shape=(batch_size, CLASS_COUNT), dtype=numpy.float)
		#y[0:batch_size, labels[sample_indices]-1] = 1.0 # Nope.  Don't get fancy.
		for i, j in zip(range(batch_size), labels[sample_indices]):
			y[i,j-1] = 1.0
		yield x, y
		#yield numpy.zeros(shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=numpy.float)

def build_conv_pool_layer(previous_input, filter_count1, pool=True):
	"""Returns a tuple of output_node, weights."""
	filter_height = 3
	filter_width = 3
	channels = previous_input.get_shape().as_list()[-1]
	max_weight = 2.0/float((filter_height*filter_width*channels)+filter_count1)
	filter1 = tf.Variable(tf.random_uniform([filter_height, filter_width, channels, filter_count1], minval=-max_weight, maxval=max_weight))
	pre_act = tf.nn.conv2d(previous_input, filter=filter1, strides=[1, 1, 1, 1], padding="SAME")
	act = activation_function(pre_act)
	print("Allocated layer with shape: {}".format(act.get_shape().as_list()))
	if pool:
		act = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
		print("Allocated layer with shape: {}".format(act.get_shape().as_list()))
	return act, filter1

def build_dense_dropout(previous_input, dropout_hook, num_units, activation=True):
	"""Returns a tuple of the output node, weights, and biases."""
	# Assume flattened.
	max_weight = 2.0/float(previous_input.get_shape().as_list()[-1]+num_units)
	weight = tf.Variable(tf.random_uniform([previous_input.get_shape().as_list()[-1], num_units], minval=-max_weight, maxval=max_weight))
	bias = tf.Variable(tf.random_uniform([num_units,], minval=-0.1, maxval=0.1))
	out = tf.nn.bias_add(tf.batch_matmul(previous_input, weight), bias)
	if activation:
		out = activation_function(out)
	if dropout_hook is not None:
		out = tf.nn.dropout(out, dropout_hook)
	print("Allocated layer with shape: {}".format(out.get_shape().as_list()))
	return out, weight, bias

def build_image_model():
	"""Yields a tuple of input, target, dropout_toggle, weights, biases."""
	# TODO: New version of tensorflow has tf.contrib.layers.flatten, max_pool2d, and convolution2d.
	output = None
	dropout_toggle = tf.placeholder(dtype=tf.float32)
	weights = list()
	biases = list()
	input_node = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
	conv, layer_weights = build_conv_pool_layer(input_node, 128, False)
	weights.append(layer_weights)
	conv, layer_weights = build_conv_pool_layer(conv, 64, True)
	weights.append(layer_weights)
	conv, layer_weights = build_conv_pool_layer(conv, 64, False)
	weights.append(layer_weights)
	conv, layer_weights = build_conv_pool_layer(conv, 64, True)
	weights.append(layer_weights)
	conv, layer_weights = build_conv_pool_layer(conv, 32, True)
	weights.append(layer_weights)

	# Flatten our convolution, preserving batches.
	flat = tf.reshape(conv, [-1, conv.get_shape()[1:].num_elements()])

	# Some FC layers.
	fc, layer_weights, layer_biases = build_dense_dropout(flat, dropout_toggle, 128)
	weights.append(layer_weights)
	biases.append(layer_biases)
	fc, layer_weights, layer_biases = build_dense_dropout(flat, dropout_toggle, 1024)
	weights.append(layer_weights)
	biases.append(layer_biases)

	# Final FC.
	out, layer_weights, layer_biases = build_dense_dropout(fc, None, CLASS_COUNT, activation=False)
	weights.append(layer_weights)
	biases.append(layer_biases)

	output = tf.nn.softmax(out)
	return input_node, output, dropout_toggle, weights, biases

def visualize_weights(weights, filename):
	from PIL import Image
	try:
		num_filters = weights.shape[-1]
		filters_per_side = int(math.ceil(math.sqrt(num_filters)))
		image_size = weights.shape[0:2]
		big_picture = numpy.zeros((image_size[0]*filters_per_side, image_size[1]*filters_per_side, 3), dtype=numpy.uint8)
		for filter in range(num_filters):
			big_picture[(filter%filters_per_side)*image_size[0]:((filter%filters_per_side)+1)*image_size[0], (filter//filters_per_side)*image_size[1]:((filter//filters_per_side)+1)*image_size[1], :] = weights[:,:,:,filter]*255
		img = Image.fromarray(numpy.asarray(big_picture, dtype=numpy.uint8))
		img.save(filename)
	except ValueError:
		pass
	except IndexError:
		pass

def main(image_path):
	ITERATIONS = 1000000
	sess = tf.Session()
	print("Loading data feed.")
	global_step = tf.Variable(0, trainable=False) # For letting our training rate fall off.
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, ITERATIONS, 0.96, staircase=True)
	#learning_rate = tf.Variable(INITIAL_LEARNING_RATE, trainable=False)
	generator = basic_image_iterator(image_path, BATCH_SIZE)
	print("Building model.")
	y = tf.placeholder(dtype=tf.float32, shape=[None, CLASS_COUNT]) # 10 classes.
	model = build_image_model()
	x, out, toggle, w, b = model
	print("Building training system.")
	loss = -tf.reduce_sum(y*tf.log(out) + (1.0-y)*tf.log(1.0-out)) 
	#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	print("Initializing variables.")
	init = tf.initialize_all_variables()
	saver = tf.train.Saver(w, b)
	sess.run(init)
	trainable_variables = tf.trainable_variables()
	print("Writing graph to disk.")
	tf.train.write_graph(sess.graph_def, '.', 'model_graph_definition.pb', as_text=False)
	print("Training.")
	avg_loss = 0
	previous_loss = 0
	running_delta_loss = 0
	increasing_loss_count = 0
	for iteration, batch in zip(range(0, ITERATIONS), generator):
		# First calculate our graient, then calculate our loss.
		#gradients, global_norm = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), 1)
		#optimizer.apply_gradients(zip(gradients, trainable_variables))
		#gradients = optimizer.compute_gradients(loss)
		#clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1) # Gradients is a list of tuples of [(tensor, variable), ...]
		#optimizer.apply_gradients(clipped_gradients) # Optimizer is expecting a list of the same.
		# Probably equal to running minimize, but lets us fiddle with the grads.

		# Loss
		_, loss_report, y_batch_display, output_display = sess.run([optimizer.minimize(loss, global_step=global_step), loss, y, out], feed_dict={x:batch[0], y:batch[1], toggle:0.5})
		#loss_report, output_display = sess.run([loss, out], feed_dict={x:batch[0], y:batch[1], toggle:0.5})

		# If we aren't getting better, adjust our learning rate.
		delta_loss = loss_report-previous_loss
		running_delta_loss = 0.99*running_delta_loss + 0.01*delta_loss
		previous_loss = loss_report
		avg_loss = 0.99*avg_loss + 0.01*loss_report

		if delta_loss >= 0: # Our threshold.
			increasing_loss_count += 1
		else:
			increasing_loss_count = 0

		if increasing_loss_count == 10:
			increasing_loss_count = 0
			lr = sess.run(learning_rate)
			running_delta_loss = 0.0
			sess.run(tf.assign(learning_rate, LEARNING_RATE_DECAY*lr))
			print("Error increasing for ten iterations.  Decreasing learning rate to {}.".format(lr*LEARNING_RATE_DECAY))
			if lr == 0:
				print("Learning rate has hit zero.")
		#elif all_negative_loss: # Always decreasing, so speed up.
		#	lr = sess.run(learning_rate)
		#	sess.run(tf.assign(learning_rate, (1.0+LEARNING_RATE_DECAY)*lr))
		
		if numpy.any(numpy.isnan(output_display)):
			print("Encountered a NaN.  Restoring from checkpoint and decreating learning rate.")
			sys.exit(-1)
			lr = sess.run(learning_rate)
			saver.restore(sess, "model_checkpoint")
			sess.run(tf.assign(learning_rate, LEARNING_RATE_DECAY*lr))
		else:	
			print("Iter {}\t|\tAvg loss {}\t|\tLast loss {}\t|\tAvg delta {}\t|\tLast delta {}".format(iteration, avg_loss, loss_report, running_delta_loss, delta_loss))
			#if iteration % 100 == 0:
			#	print("Prediction:\n{}".format(output_display))
			if iteration % SAVE_INTERVAL == 0:
				saver.save(sess, "model_checkpoint")
				#weight_matrix = sess.run(w[0])
				#visualize_weights(weight_matrix, "weights_layer_{}_iter_{}.png".format(0, iteration))

if __name__=="__main__":
	from IPython.core.debugger import Tracer
	Tracer()()
	main(sys.argv[1])
