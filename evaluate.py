from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import csv 

def counts(y_pred, y_true, threshold = 0.5):
	""" 
	Inputs:
		 y_true - true labels (np.array(rows: #examples; cols: 1))
		 y_pred - predicted labels (np.array(rows: #examples; cols: 1))
		 threshold - decision threshold (float)

	Outputs:
		tp - true positives
		tn - true negatives
		fp - false positives
		fn - false negatives

	"""

	tp = 0
	fp = 0
	tn = 0
	fn = 0
	np.reshape(y_pred, (-1,))
	np.reshape(y_true, (-1,))
	m = np.shape(y_true)[0]
	for i in range(m):
		if y_pred[i] >= threshold:
			if y_true[i] == 1.0:
				tp += 1
			else:
				fn += 1
		else:
			if y_true [i] == 1.0:
				fp += 1
			else:
				tn += 1

	return tp,fn,fp,tn

def stats(tp,fn,fp,tn):
	""" 
	Inputs:
		tp - true positives
		tn - true negatives
		fp - false positives
		fn - false negatives

	Outputs:
		accuracy,precision,sensitivity,specificity,F1

	"""
	eps = 1e-7
	accuracy =  (tp+tn)/(tp+fn+fp+tn)
	precision = tp/(tp+fp+eps)
	sensitivity = tp/(tp+fn+eps) # = recall = tp/(tp+fn+eps)
	specificity = tn/(tn+fp+eps)
	F1 = (2/((1/(sensitivity + eps)) +  (1/(precision+eps))))

	return accuracy,precision,sensitivity,specificity,F1

def ROCandAUROC(y_pred,y_true,jpeg_path = 'ROC.jpeg',csv_path = 'ROC.csv'):

	#getting true positive rate and false positive rate
	fpr, tpr, thresholds = roc_curve(y_true, y_pred)

	# Plot ROC Curve
	plt.figure()
	plt.plot([0, 1], ls="--")
	plt.plot([0, 0], [1, 0] , c=".7")
	plt.plot([1, 1] , c=".7")
	plt.plot(fpr, tpr)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(jpeg_path)

	auc = roc_auc_score(y_true, y_pred)

	#finding best threshold
	p = fpr*np.flip(tpr)
	indx_best = 0
	for i in range(len(p)):
		if p[indx_best] <= p[i]:
			indx_best = i


	#saving data to csv
	thresholds = np.reshape(thresholds,(thresholds.shape[0],1))
	fpr = np.reshape(fpr,(fpr.shape[0],1))
	tpr = np.reshape(tpr,(tpr.shape[0],1)) 
	data = np.concatenate((thresholds,fpr,tpr),axis=1)
	np.savetxt(csv_path,data,delimiter=',',header='thresholds,fpr,tpr')

	return auc, thresholds[indx_best]

def find_best_threshold(y_pred,y_true):
	num_thresholds = 1000
	thresholds = np.zeros((num_thresholds,1))
	accuracy = np.zeros((num_thresholds,1))
	for i in range(num_thresholds):
		thresholds[i] = i*(1/num_thresholds)
		accuracy[i] = np.sum(1.0*(1.0*(y_pred > thresholds[i]) == y_true))/y_true.shape[0]
	return thresholds[np.argmax(accuracy)]