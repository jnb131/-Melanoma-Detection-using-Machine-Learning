import evaluate
import img_proc
import argparse
import csv
import numpy as np
from pathlib import Path
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from kt_utils import *
import tensorflow as tf
from tensorflow import keras

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

def CNN(data_gen,epochs = 10):
    """
   
    """

    X_input = Input(data_gen.example_shape_tensor()) #shape: 224x224x3

    X = ZeroPadding2D((3, 3))(X_input) #shape: 230x230x3
    X = Conv2D(filters=32, kernel_size=(7, 7), strides = (1, 1), name = 'conv0')(X) #shape: 224x224x32
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool0')(X) #shape: 112x112x32

    X = ZeroPadding2D((2, 2))(X_input) #shape: 116x116x32
    X = Conv2D(filters=64, kernel_size=(5, 5), strides = (1, 1), name = 'conv1')(X) #shape: 112x112x64
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool1')(X) #shape: 56x56x64

    X = ZeroPadding2D((1, 1))(X_input) #shape: 58x58x64
    X = Conv2D(filters=128, kernel_size=(3, 3), strides = (1, 1), name = 'conv2')(X) #shape: 56x56x128
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2),name='max_pool2')(X) #shape: 28x28x128

    X = Conv2D(filters=128, kernel_size=(3, 3), strides = (3, 3), name = 'conv3')(X) #shape: 9x9x128
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(3,3),name='max_pool3')(X) #shape: 3x3x128


    X = Flatten()(X) #shape: 1152
    X = Dense(units=512, activation='relu', name='fc0')(X)
    X = Dense(units=32, activation='relu', name='fc1')(X)
    X = Dense(units=1, activation='sigmoid', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='CNN') # Total number of trainable params = 737,537
    model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
    model.fit(data_gen, epochs = epochs, use_multiprocessing=True, workers=8)

    return model

def main(data_dir):
    print('---------- Loading Training Set ----------')
    BATCH_SIZE = 100
    data_gen_train = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=False)
    # X_train, y_train = data_train.__getitem__(1)
    # print(f"X_train.shape = {X_train.shape}")
    # print(f"y_train.shape = {y_train.shape}")

    # print('---------- Training Model ----------')
    # model = CNN(data_gen_train, epochs = 20)
    # model.save('savedCNN_' + str(data_dir))

    print('---------- Loading Model Set ----------')
    model = keras.models.load_model('savedCNN_' + str(data_dir))

    print('---------- Predicting on Training Set ----------')
    data_gen_train_test = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=False, flatten=False)
    y_train = data_gen_train_test.get_labels()
    y_train_pred = model.predict(data_gen_train_test)

    print('---------- Predicting on Validation Set ----------')
    data_gen_valid = img_proc.Data_Generator(data_dir / 'valid', BATCH_SIZE, shuffle=False, flatten=False)
    y_valid = data_gen_valid.get_labels()
    y_valid_pred = model.predict(data_gen_valid)

    print('---------- Predicting on Test Set ----------')
    data_gen_test = img_proc.Data_Generator(data_dir / 'test', BATCH_SIZE, shuffle=False, flatten=False)
    y_test = data_gen_test.get_labels()
    y_test_pred = model.predict(data_gen_test)
    
    #saving data to csv
    print('---------- Saving Predictions to csv ----------')
    data_train = np.concatenate((y_train,y_train_pred),axis=1)
    data_valid = np.concatenate((y_valid,y_valid_pred),axis=1)
    data_test = np.concatenate((y_test,y_test_pred),axis=1)
    np.savetxt('predictions_train_cnn.csv',data_train,delimiter=',',header='y_train,y_train_pred')
    np.savetxt('predictions_valid_cnn.csv',data_valid,delimiter=',',header='y_valid,y_valid_pred')
    np.savetxt('predictions_test_cnn.csv',data_test,delimiter=',',header='y_tes,y_test_pred')

    #calculating metrics
    print('---------- Calculating Threshold and ROC ----------')
    threshold_best_accuracy = evaluate.find_best_threshold(y_valid_pred,y_valid)
    auc_roc_train,threshold_best = evaluate.ROCandAUROC(y_train_pred,y_train,'ROC_train_data_cnn.jpeg', 'ROC_train_data_cnn.csv')
    auc_roc_valid,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_cnn.jpeg', 'ROC_valid_data_cnn.csv')
    auc_roc_test,threshold_best = evaluate.ROCandAUROC(y_test_pred,y_test,'ROC_test_data_cnn.jpeg', 'ROC_test_data_cnn.csv')

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