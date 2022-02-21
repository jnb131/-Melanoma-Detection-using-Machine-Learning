import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from pathlib import Path
import argparse

import img_proc
import evaluate


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

def transfer_learning(data_gen, base_model, epochs=20):
    X_input = Input(data_gen.example_shape_tensor()) #shape: 224x224x3

    X = base_model(X_input)
    
    # X = ZeroPadding2D((3, 3))(X) #shape: 13x13x2048
    # X = Conv2D(filters=2048, kernel_size=(7, 7), strides = (1, 1), name = 'conv0')(X) #shape: 7x7x2048
    # X = BatchNormalization(axis = 3, name = 'bn0')(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D(pool_size=(3, 3), strides=(2,2),name='max_pool0')(X) #shape: 3x3x2048

    # X = ZeroPadding2D((1, 1))(X) #shape: 5x5x2048
    # X = Conv2D(filters=4096, kernel_size=(3, 3), strides = (1, 1), name = 'conv1')(X) #shape: 3x3x4096
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D(pool_size=(3, 3), strides=(2,2),name='max_pool1')(X) #shape: 2x2x4096

    X = Flatten()(X) #shape: 16,384
    X = Dense(units=4096, activation='relu', name='fc0')(X)
    X = Dense(units=1024, activation='relu', name='fc1')(X)
    X = Dense(units=256, activation='relu', name='fc2')(X)
    X = Dense(units=32, activation='relu', name='fc3')(X)
    X = Dense(units=1, activation='sigmoid', name='fc4')(X)

    model = Model(inputs = X_input, outputs = X, name='CNN') # Total number of trainable params = 737,537
    model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
    model.fit(data_gen, epochs = epochs, use_multiprocessing=True, workers=8)

    return model

def main(data_dir,BATCH_SIZE=100):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    
    print('---------- Loading Training Set ----------')
    feats = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=False)

    print('---------- Training Model ----------')
    model = transfer_learning(feats, base_model, epochs=5)

    print('---------- Saving Model ----------')
    model.save('saved_transfer_model')

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
    np.savetxt('predictions_train_transfer.csv',data_train,delimiter=',',header='y_train,y_train_pred')
    np.savetxt('predictions_valid_transfer.csv',data_valid,delimiter=',',header='y_valid,y_valid_pred')
    np.savetxt('predictions_test_transfer.csv',data_test,delimiter=',',header='y_tes,y_test_pred')

    #calculating metrics
    print('---------- Calculating Threshold and ROC ----------')
    threshold_best_accuracy = evaluate.find_best_threshold(y_valid_pred,y_valid)
    auc_roc_train,threshold_best = evaluate.ROCandAUROC(y_train_pred,y_train,'ROC_train_data_transfer.jpeg', 'ROC_train_data_transfer.csv')
    auc_roc_valid,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_transfer.jpeg', 'ROC_valid_data_transfer.csv')
    auc_roc_test,threshold_best = evaluate.ROCandAUROC(y_test_pred,y_test,'ROC_test_data_transfer.jpeg', 'ROC_test_data_transfer.csv')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    main(data_dir)