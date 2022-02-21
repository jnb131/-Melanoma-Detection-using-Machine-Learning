from random import shuffle
import numpy as np
import img_proc
import evaluate


def main():
    """Problem: Logistic regression
    """

    step_size = 0.1
    TRAIN_DATA_BATCH_SIZE = 3000
    BATCH_SIZE = 500
    TRAIN_DATA_DIR = 'data_3000'
    data_dir = 'data'
    data_gen_train = img_proc.Data_Generator(TRAIN_DATA_DIR + '/train_sep', TRAIN_DATA_BATCH_SIZE, shuffle=True,
                                             flatten=True)
    x_train, y_train = data_gen_train.__getitem__(0)
    print("mini batch")
    model = LogisticRegression(step_size=step_size)
    model.fit_mini_batch(x_train, y_train, BATCH_SIZE)
    # model.fit(x_train, y_train)
    model.save_weights()

    print('---------- Predicting on Training Set ----------')
    y_train_pred = model.predict(x_train)

    print('---------- Predicting on Validation Set ----------')
    data_gen_valid = img_proc.Data_Generator('data/valid', BATCH_SIZE, shuffle=False, flatten=True)
    y_valid = data_gen_valid.get_labels()
    y_valid_pred = model.predict_from_gen(data_gen_valid)

    print('---------- Predicting on Test Set ----------')
    data_gen_test = img_proc.Data_Generator('data/test', BATCH_SIZE, shuffle=False, flatten=True)
    y_test = data_gen_test.get_labels()
    y_test_pred = model.predict_from_gen(data_gen_test)

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



class LogisticRegression:

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def save_weights(self):
        np.savetxt("log_reg_weights.csv", self.theta, delimiter=",")

    def sigmoid_theta(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.theta)))

    def fit(self, x, y):
        """

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if not self.theta:
            self.theta = np.zeros(np.shape(x)[1])
        for i in range(self.max_iter):
            addition = np.zeros(len(x[0]))
            for j in range(len(x)):
                addition += x[j] * (y[j] - self.sigmoid_theta(x[j]))
            self.theta += self.step_size * addition
            error = np.linalg.norm(addition * self.step_size)
            if (error < self.eps):
                return

    def fit_mini_batch(self, x, y, batch_size):
        if self.theta == None:
            self.theta = np.zeros(np.shape(x)[1])

        for i in range(self.max_iter):
            for k in range(int(len(x) / batch_size)):
                addition = np.zeros(len(x[0]))
                for j in range(batch_size):
                    addition += x[k * batch_size + j] * (
                                y[k * batch_size + j] - self.sigmoid_theta(x[k * batch_size + j]))
                self.theta += self.step_size * addition
                error = np.linalg.norm(addition * self.step_size)
                if (error < self.eps):
                    print(f"It took {i} iterations with step size {self.step_size}")
                    return


    def predict_from_gen(self, data_gen):
        outputs = []

        for batch in range(data_gen.__len__()):
            x_batch, y_batch = data_gen.__getitem__(batch)
            predicted_probability_batch = self.sigmoid_theta(x_batch)
            outputs_batch = predicted_probability_batch
            for i in range(outputs_batch.shape[0]):
                outputs.append(outputs_batch[i])

        return np.array(outputs)

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return self.sigmoid_theta(x)


if __name__ == '__main__':
    main()