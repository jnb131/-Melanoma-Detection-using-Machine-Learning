import numpy as np
import tensorflow as tf
from tensorflow import keras
import img_proc
import evaluate
from pathlib import Path
from DLNN import DLNN

# def extract_features(base_model, train_data):
#     for sample in train_data:
#         print(sample)
#         print("Before")
#         features = base_model.predict(train_data, steps=100)
#         # TODO: normalize and flatten features vector
#         print("HELLO")
#         print(np.shape(features), np.shape(sample[0]), np.shape(sample[1]))
#         yield (features, sample[1])

def transfer(data_dir, BATCH_SIZE=100):
    print('---------- Loading Training Set ----------')
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False)
    feats = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=False, model=base_model, flatten_post_model=True)
    
    print('---------- Training DLNN ----------')
    model = DLNN(feats, [1024, 256, 64, 16, 4, 1], epochs=1)

    print('---------- Saving Model ----------')
    model.save('saved_DLNN_transfer')

    # print('---------- Predicting on Training Set ----------')
    # data_gen_train_test = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=False, flatten=False, model=base_model, flatten_post_model=True)
    # y_train = data_gen_train_test.get_labels()
    # y_train_pred = model.predict(data_gen_train_test)

    # print('---------- Predicting on Validation Set ----------')
    # data_gen_valid = img_proc.Data_Generator(data_dir / 'valid', BATCH_SIZE, shuffle=False, flatten=False, model=base_model, flatten_post_model=True)
    # y_valid = data_gen_valid.get_labels()
    # y_valid_pred = model.predict(data_gen_valid)

    # print('---------- Predicting on Test Set ----------')
    # data_gen_test = img_proc.Data_Generator(data_dir / 'test', BATCH_SIZE, shuffle=False, flatten=False, model=base_model, flatten_post_model=True)
    # y_test = data_gen_test.get_labels()
    # y_test_pred = model.predict(data_gen_test)
    
    # #saving data to csv
    # print('---------- Saving Predictions to csv ----------')
    # data_train = np.concatenate((y_train,y_train_pred),axis=1)
    # data_valid = np.concatenate((y_valid,y_valid_pred),axis=1)
    # data_test = np.concatenate((y_test,y_test_pred),axis=1)
    # np.savetxt('predictions_train_transfer.csv',data_train,delimiter=',',header='y_train,y_train_pred')
    # np.savetxt('predictions_valid_transfer.csv',data_valid,delimiter=',',header='y_valid,y_valid_pred')
    # np.savetxt('predictions_test_transfer.csv',data_test,delimiter=',',header='y_tes,y_test_pred')

    # #calculating metrics
    # print('---------- Calculating Threshold and ROC ----------')
    # threshold_best_accuracy = evaluate.find_best_threshold(y_valid_pred,y_valid)
    # auc_roc_train,threshold_best = evaluate.ROCandAUROC(y_train_pred,y_train,'ROC_train_data_transfer.jpeg', 'ROC_train_data_transfer.csv')
    # auc_roc_valid,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_transfer.jpeg', 'ROC_valid_data_transfer.csv')
    # auc_roc_test,threshold_best = evaluate.ROCandAUROC(y_test_pred,y_test,'ROC_test_data_transfer.jpeg', 'ROC_test_data_transfer.csv')

    # print('---------- Calculating Metrics on Train, Validation and Test ----------')
    # tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best_accuracy) #threshold_best)
    # acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    # print("\nStats for predictions on train set:")
    # print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
    # print(f"Accuracy = {acc}")
    # print(f"Precision = {prec}")
    # print(f"Sensitivity = {sens}")
    # print(f"Specificity = {spec}")
    # print(f"F1 score = {F1}")
    # print(f"AUCROC = {auc_roc_train}")


    # tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best_accuracy) #threshold_best)
    # acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    # print("\nStats for predictions on validation set:")
    # print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
    # print(f"Accuracy = {acc}")
    # print(f"Precision = {prec}")
    # print(f"Sensitivity = {sens}")
    # print(f"Specificity = {spec}")
    # print(f"F1 score = {F1}")
    # print(f"AUCROC = {auc_roc_valid}")

    # tp,fn,fp,tn = evaluate.counts(y_test_pred, y_test, threshold = threshold_best_accuracy) #threshold_best)
    # acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    # print("\nStats for predictions on test set:")
    # print(f"Threshold = {threshold_best_accuracy}") #threshold_best}")
    # print(f"Accuracy = {acc}")
    # print(f"Precision = {prec}")
    # print(f"Sensitivity = {sens}")
    # print(f"Specificity = {spec}")
    # print(f"F1 score = {F1}")
    # print(f"AUCROC = {auc_roc_test}")
def main():
    BATCH_SIZE = 100
    DATA_DIR = Path('data_200')
    transfer(DATA_DIR, BATCH_SIZE)

if __name__ == "__main__":
    main()


