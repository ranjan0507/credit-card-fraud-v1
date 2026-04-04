from src.data_features.data_loader import load_data
from src.data_features.preprocessing import split_data
from src.pipelines.pipelines import get_pipeline
from sklearn.metrics import classification_report , roc_auc_score , precision_recall_curve , auc , f1_score , average_precision_score
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def get_best_threshold(y_val,y_probs):
	thresholds = np.linspace(0,1,100)
	
	best_t=0
	best_f1=0

	for t in thresholds:
		y_pred = (y_probs>=t).astype(int)
		f1=f1_score(y_val,y_pred)

		if f1>best_f1:
			best_t=t
			best_f1=f1
	
	return best_t,best_f1

def cross_validated_thresholds(X,y,model):
	skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
	thresholds=[]
	f1_scores=[]

	for fold , (train_idx , val_idx) in enumerate(skf.split(X,y)):
		print(f"Fold { fold+1}")
		X_train,X_val = X.iloc[train_idx] , X.iloc[val_idx]
		y_train,y_val = y.iloc[train_idx] , y.iloc[val_idx]
		pipeline = get_pipeline(model)
		pipeline.fit(X_train,y_train)

		y_probs = pipeline.predict_proba(X_val)[:,1]

		best_t , best_f1 = get_best_threshold(y_val,y_probs)

		print(f"Best threshold : {best_t} , Best F1 : {best_f1}")

		thresholds.append(best_t)
		f1_scores.append(best_f1)
	 
	return thresholds , f1_scores


def run_training_pipeline(data_path,model_path):
	df=load_data(data_path)

	X_train,y_train,X_test,y_test = split_data(df)

	base_model = RandomForestClassifier(n_jobs=-1,random_state=42,n_estimators=100)
	model = CalibratedClassifierCV(estimator=base_model,method='sigmoid',cv=3)

	thresholds , f1_scores = cross_validated_thresholds(X_train,y_train,model)
	best_t = np.mean(thresholds)

	print("Thresholds:", thresholds)
	print("Threshold mean:", np.mean(thresholds))
	print("Threshold std:", np.std(thresholds))

	print("F1 scores:", f1_scores)
	print("Mean F1:", np.mean(f1_scores))
	print("Std F1:", np.std(f1_scores))

	print(f"Best threshold : {best_t}")

	pipeline = get_pipeline(model)
	pipeline.fit(X_train,y_train)

	y_test_probs = pipeline.predict_proba(X_test)[:, 1]
	y_test_pred = (y_test_probs >= best_t).astype(int)

	print(classification_report(y_test,y_test_pred))

	precision , recall , thresholds = precision_recall_curve(y_test,y_test_probs)
	pr_auc = auc(recall,precision)
	print("PR-AUC: ", pr_auc) 

	print(f"ROC_AUC : ",roc_auc_score(y_test,y_test_probs))

	joblib.dump({"model":pipeline,"threshold":best_t}
		,f"{model_path}/model_pipeline.pkl")

