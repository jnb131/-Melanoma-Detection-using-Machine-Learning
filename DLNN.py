import evaluate
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
# from keras.utils.vis_utils import plot_model
import img_proc
from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow import keras

def DLNN(data_gen,nn_dims,epochs = 20,batch_size = 25,loss='binary_crossentropy',lambd = 1e-4):
	""" 
	Inputs:
		 data_gen - img_proc.Data_Generator that produces (x, y) training data in batches
		 nn_dims - list with number of neurons in each layer [#input layer, # 1st hidden layer, ..., 1 neuron with sigmoid]
		 epochs - number of iterations to run gradient descent 
		 batch_size - number of training examples to include in a batch
		 loss - loss function for gradient descent to optimize over (input as string)

	Outputs:
		trained_model - model trained in keras

	"""
	n = data_gen.num_features_flat()
	layers = len(nn_dims) # number of layers

	trained_model = Sequential()
	trained_model.add(Dense(nn_dims[0], input_dim = n, activation = 'relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	for i in range(1,layers - 1):
		neurons = nn_dims[i]
		trained_model.add(Dense(neurons, activation='relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros', kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	trained_model.add(Dense(1, activation='sigmoid',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))
	print(trained_model.summary())
	trained_model.compile(loss = loss, optimizer = 'adam', metrics=['accuracy'])
	trained_model.fit(tf.data.Dataset.from_generator(data_gen.__getitem__,output_types = tf.float16), epochs = epochs, use_multiprocessing=True, workers=8)

	return trained_model

def main(data_dir):
	print('---------- Loading Training Data ----------')
	BATCH_SIZE = 100
	data_gen_train = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=True)

	# print('---------- Training Model ----------')
	# model = DLNN(data_gen_train,[1024,256,64,16,4,1],epochs = 20)

	# print('---------- Saving Model ----------')
	# model.save('savedDNN_' + str(data_dir))
	print('---------- Loading Model Set ----------')
	model = keras.models.load_model('savedDNN_' + str(data_dir))

	print('---------- Predicting on Training Set ----------')
	data_gen_train_test = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=False, flatten=True)
	y_train = data_gen_train_test.get_labels()
	y_train_pred = model.predict(data_gen_train_test)

	print('---------- Predicting on Validation Set ----------')
	data_gen_valid = img_proc.Data_Generator(data_dir / 'valid', BATCH_SIZE, shuffle=False, flatten=True)
	y_valid = data_gen_valid.get_labels()
	y_valid_pred = model.predict(data_gen_valid)

	print('---------- Predicting on Test Set ----------')
	data_gen_test = img_proc.Data_Generator(data_dir / 'test', BATCH_SIZE, shuffle=False, flatten=True)
	y_test = data_gen_test.get_labels()
	y_test_pred = model.predict(data_gen_test)

	#saving data to csv
	print('---------- Saving Predictions to csv ----------')
	data_train = np.concatenate((y_train,y_train_pred),axis=1)
	data_valid = np.concatenate((y_valid,y_valid_pred),axis=1)
	data_test = np.concatenate((y_test,y_test_pred),axis=1)
	np.savetxt('predictions_train_dlnn.csv',data_train,delimiter=',',header='y_train,y_train_pred')
	np.savetxt('predictions_valid_dlnn.csv',data_valid,delimiter=',',header='y_valid,y_valid_pred')
	np.savetxt('predictions_test_dlnn.csv',data_test,delimiter=',',header='y_tes,y_test_pred')

	#calculating metrics
	print('---------- Calculating Threshold and ROC ----------')
	threshold_best_accuracy = evaluate.find_best_threshold(y_valid_pred,y_valid)
	auc_roc_train,threshold_best = evaluate.ROCandAUROC(y_train_pred,y_train,'ROC_train_data_dlnn.jpeg', 'ROC_train_data_dlnn.csv')
	auc_roc_valid,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_dlnn.jpeg', 'ROC_valid_data_dlnn.csv')
	auc_roc_test,threshold_best = evaluate.ROCandAUROC(y_test_pred,y_test,'ROC_test_data_dlnn.jpeg', 'ROC_test_data_dlnn.csv')

	print('---------- Calculating Metrics on Train, Validation and Test ----------')
	tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best_accuracy) #threshold_best)
	acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
	print("\nStats for predictions on train set:")
	print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Sensitivity = {sens}")
	print(f"Specificity = {spec}")
	print(f"F1 score = {F1}")
	print(f"AUCROC = {auc_roc_train}")

	tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best_accuracy) #threshold_best)
	acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
	print("\nStats for predictions on validation set:")
	print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Sensitivity = {sens}")
	print(f"Specificity = {spec}")
	print(f"F1 score = {F1}")
	print(f"AUCROC = {auc_roc_valid}")

	tp,fn,fp,tn = evaluate.counts(y_test_pred, y_test, threshold = threshold_best_accuracy) #threshold_best)
	acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
	print("\nStats for predictions on test set:")
	print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Sensitivity = {sens}")
	print(f"Specificity = {spec}")
	print(f"F1 score = {F1}")
	print(f"AUCROC = {auc_roc_test}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', default='data')
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	main(data_dir)

