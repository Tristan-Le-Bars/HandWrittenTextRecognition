import tensorflow as tf
import numpy as np
import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from Preprocessor import preprocess
import matplotlib.pyplot as plt
from datetime import datetime


class FilePaths:
	# all the path and filename needed
	charList = '../model/charList.txt'
	train = '../data/'
	testImg = '../Picture/test.png'


def train(model, loader):
	# train the neural network
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch) # print the training number

		# train
		loader.trainSet() # load a training set
		while loader.hasNext(): # while the training set is not over
			iterInfo = loader.getIteratorInfo() # get the iterator and the size of the training set
			batch = loader.getNext() # get the next image and its groud truth
			loss = model.trainBatch(batch) # train the neural network and get its loss value
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = evaluation(model, loader)[0]
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('error rate reduced, updating model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. More training is useless.' % earlyStopping)
			break


def evaluation(model, loader):
	print('Model evaluation')
	loader.validationSet()
	numCharErr = 0 # total number of error
	numCharTotal = 0 # total number of character
	numGoodWord = 0 # total number of good word
	numWordTotal = 0 # total number of word
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Set:', iterInfo[0],'/', iterInfo[1]) # print the set currently training
		batch = loader.getNext()
		(recognized, _) = model.makePrediction(batch)
		for i in range(len(recognized)):
			numGoodWord += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			mistake = editdistance.eval(recognized[i], batch.gtTexts[i]) # check the number of mistake in the prediction
			numCharErr += mistake
			numCharTotal += len(batch.gtTexts[i])
			print('succes: ' if mistake == 0 else 'failure, %d mistake(s):' % mistake,'label = "' + batch.gtTexts[i] + '" ', '->', ' prediction = "' + recognized[i] + '"') # print the comparaison of the label and the prediction
	
	# print final prediciton result
	charErrorRate = numCharErr / numCharTotal # compute the error rate
	wordAccuracy = numGoodWord / numWordTotal # compute the accuracy
	print("Error rate of character prediciton : %f%%" % (charErrorRate*100.0))
	print("Accuracy of words prediction : %f%%" % (wordAccuracy*100.0))
	return charErrorRate, wordAccuracy

def plot_value_array(title, valueTuple, nameTuple):
	#setup the graph of the pyplot window
    plt.title("Model data")

    y_pos = np.arange(len(nameTuple))
    plt.xticks(y_pos, nameTuple)
    plt.ylabel("Percentage")
    plt.yticks([0,10,20,30,40,50, 60,70,80,90,100])

    bar1 = plt.bar([0], valueTuple[0] * 100)
    bar1[int(valueTuple[0])].set_color('red')
    
    bar2 = plt.bar([1], valueTuple[1] * 100)
    bar2[int(valueTuple[1])].set_color('blue')

    for rect in bar1 + bar2:
    	height = rect.get_height()
    	plt.text(rect.get_x() + rect.get_width()/2.0, height, "~" + str(int(height)) + '%', ha='center', va='bottom')

    plt.ylim([0, 100])

def displayPlot(charErrorRate, wordAccuracy, recognized, probability, img):
	#display a pyplotvwindow to visualize the performance of the AI
	dataTuple = []
	dataTuple.append(charErrorRate)
	dataTuple.append(wordAccuracy)

	dataNames = []
	dataNames.append("|Characters prediction error rate|")
	dataNames.append("|Words prediction accuracy|")

	plt.figure(figsize=(12,5))
	plt.subplot(1,2,1)
	plt.imshow(img)
	str1 = "Predicted word: " + "\"" + recognized[0] + "\""
	plt.text(-100, 5, str1)
	plt.subplot(1,2,2)
	plot_value_array("Model data", dataTuple, dataNames)

	plt.grid(False)
	plt.show()

def predictTest(model, fnImg, charErrorRate, wordAccuracy):
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize) # preprocess the image
	batch = Batch(None, [img]) # create a batch object
	(recognized, probability) = model.makePrediction(batch, True)
	print('Predicted word:', '"' + recognized[0] + '"')
	now = datetime.now().time()
	current_time = now.strftime("%H:%M:%S")
	text_file = open("../predictions.txt", "a")
	message = current_time + " >>> " + "prediction : " + str(recognized[0]) + " | model accuracy : " + str(wordAccuracy) + '\n'
	print(message, file=text_file)
	text_file.close()
	displayPlot(charErrorRate, wordAccuracy, recognized, probability, img)


def main():
	tf.compat.v1.disable_eager_execution()
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true') # train the AI
	parser.add_argument('--make_test', help='validate the NN', action='store_true') # make a test of the AI

	args = parser.parse_args()

	decoderType = DecoderType.BestPath

	# train or validate on IAM dataset	
	if args.train or args.make_test:
		# load training data, create tensorflow model
		loader = DataLoader(FilePaths.train, Model.batchSize, Model.imgSize, Model.maxTextLen)

		open(FilePaths.charList, 'w').write(str().join(loader.charList)) # save characters of model for inference mode
		
		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.make_test:
			model = Model(loader.charList, decoderType, mustRestore=True)
			charErrorRate, wordAccuracy = evaluation(model, loader)
			predictTest(model, FilePaths.testImg, charErrorRate, wordAccuracy)

	# infer text on test image
	else:
		loader = DataLoader(FilePaths.train, Model.batchSize, Model.imgSize, Model.maxTextLen)
		model = Model(loader.charList, decoderType)
		train(model, loader)
		predictTest(model, FilePaths.testImg, charErrorRate, wordAccuracy)

if __name__ == '__main__':
	main()