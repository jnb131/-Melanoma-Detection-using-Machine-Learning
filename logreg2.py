import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import img_proc
import evaluate


def main():
    step_size = 0.01
    TRAIN_DATA_BATCH_SIZE = 3000
    data_dir = 'data'
    BATCH_SIZE = 500

    data_gen_train = img_proc.Data_Generator(data_dir + '/train_sep', TRAIN_DATA_BATCH_SIZE, shuffle=True,
                                                 flatten=True)
    x_train, y_train = data_gen_train.__getitem__(0)

    model = Sequential({
        Dense(1, input_shape=(150528,), activation='sigmoid')
    })
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

    model.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        epochs=10,
        batch_size=500
    )

    print('---------- Predicting on Training Set ----------')
    y_train_pred = model.predict(x_train)

    print('---------- Predicting on Validation Set ----------')
    data_gen_valid = img_proc.Data_Generator('data/valid', BATCH_SIZE, shuffle=False, flatten=True)
    y_valid = data_gen_valid.get_labels()
    y_valid_pred = model.predict(data_gen_valid)

    print('---------- Predicting on Test Set ----------')
    data_gen_test = img_proc.Data_Generator('data/test', BATCH_SIZE, shuffle=False, flatten=True)
    y_test = data_gen_test.get_labels()
    y_test_pred = model.predict(data_gen_test)

    # calculating metrics
    print('---------- Calculating Threshold and ROC ----------')
    threshold_best_accuracy = evaluate.find_best_threshold(y_valid_pred, y_valid)
    auc_roc_train, threshold_best = evaluate.ROCandAUROC(y_train_pred, y_train, 'ROC_train_data_logreg.jpeg',
                                                         'ROC_train_data_logreg.csv')
    auc_roc_valid, threshold_best = evaluate.ROCandAUROC(y_valid_pred, y_valid, 'ROC_valid_data_logreg.jpeg',
                                                         'ROC_valid_data_logreg.csv')
    auc_roc_test, threshold_best = evaluate.ROCandAUROC(y_test_pred, y_test, 'ROC_test_data_logreg.jpeg',
                                                        'ROC_test_data_logreg.csv')

    print('---------- Calculating Metrics on Train, Validation and Test ----------')
    tp, fn, fp, tn = evaluate.counts(y_train_pred, y_train)  # threshold_best)
    acc, prec, sens, spec, F1 = evaluate.stats(tp, fn, fp, tn)
    print("\nStats for predictions on train set:")
    print(f"Threshold = {threshold_best_accuracy}")  # threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")
    print(f"AUCROC = {auc_roc_train}")

    tp, fn, fp, tn = evaluate.counts(y_valid_pred, y_valid)  # threshold_best)
    acc, prec, sens, spec, F1 = evaluate.stats(tp, fn, fp, tn)
    print("\nStats for predictions on validation set:")
    print(f"Threshold = {threshold_best_accuracy}")  # threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")
    print(f"AUCROC = {auc_roc_valid}")

    tp, fn, fp, tn = evaluate.counts(y_test_pred, y_test)  # threshold_best)
    acc, prec, sens, spec, F1 = evaluate.stats(tp, fn, fp, tn)
    print("\nStats for predictions on test set:")
    print(f"Threshold = {threshold_best_accuracy}")  # threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")
    print(f"AUCROC = {auc_roc_test}")


if __name__ == '__main__':
    main()