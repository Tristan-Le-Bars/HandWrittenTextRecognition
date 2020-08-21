import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	# tf model for htr

	# model constants
	batchSize = 50 # size of the set
	imgSize = (128, 32) # size of the image
	maxTextLen = 32 # maximum text size

    # INITIALIZE THE MODEL 
	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0

		# Whether to use normalization over a batch or a population
		self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

		# input image batch
		self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1])) # create a tensor with the shape of an image

		# setup CNN, RNN and CTC

        ###########################################
        # setup of all the neural networks layers #
        ###########################################
		self.setupCNN() # convolutional layer
		self.setupRNN() # recurrence layer
		self.setupCTC() # connectionist temporal classification layer

		# setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        # make a collection of operations performed when the graph run
		self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(self.update_ops): # with ~= try 
			self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss) # optimizer that implements the RMSProp algorithm

		(self.sess, self.saver) = self.setupTF() # initialize tensorflow

			
	def setupCNN(self):
		# create CNN layers and return output of these layers
		cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3) # add a dimention to the input image tensor

        #####################################
		# list of parameters for the layers #
        #####################################
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			kernel = tf.Variable(tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1)) # ??????
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1)) # create the convolutional layer
			conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train) # normalization of values
			relu = tf.nn.relu(conv_norm) # compute rectified linear
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID') # max pooling of the values

		self.cnnOut4d = pool


	def setupRNN(self):
		# create RNN layers and return output of these layers
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2]) # remove one dimension from a tensor

		numHidden = 256
		cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # create 2 layers of 256 recurrent network cell

		stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True) # RNN cell composed sequentially of multiple simple cells

		# bidirectional RNN
		((fw, bw), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype) # creates a dynamic version of bidirectional recurrent neural network
									
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2) # adding dimensions
									
		kernel = tf.Variable(tf.random.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self):
		# create CTC loss and decoder and return them

		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2]) # make a "fusion" of 2 arrays
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]) , tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))

		# calculate loss for batch
		self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# calculate loss for each element to compute label probability
		self.savedCtcInput = tf.compat.v1.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True) # compute ctc loss

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen) # performs greedy decoding on the logits given in input (best path)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False) # performs beam search decoding on the logits given in input
	

	def setupTF(self):
		# initialize TF

		sess=tf.compat.v1.Session() # TF session

		saver = tf.compat.v1.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # check for the filename of latest saved checkpoint file

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			print('Can\'t find any saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Initialization with the stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Initialization with new values')
			sess.run(tf.compat.v1.global_variables_initializer()) # initialize new variables

		return (sess,saver)


	def toSparse(self, texts):
		# put labels texts into sparse tensor for ctc_loss
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		# extract texts from output of CTC decoder
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		decoded=ctcOutput[0][0] 

		# go over all indices and save mapping: batch -> values
		idxDict = { b : [] for b in range(batchSize) }
		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0]
			encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		# feed a batch into the neural network to train it
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal

	def makePrediction(self, batch, calcProbability=False, probabilityOfGT=False):
		# feed a batch into the neural network to recognize the texts
		
		numBatchElements = len(batch.imgs) # number of elements in the set
		evalRnnOutput = calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(lossVals)

		return (texts, probs)
	

	def save(self):
		# save the model to file
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
