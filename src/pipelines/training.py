from src.data_features.data_loader import load_data
from src.data_features.preprocessing import split_data
from src.pipelines.pipelines import get_pipeline
from sklearn.metrics import classification_report , roc_auc_score , precision_recall_curve , auc , recall_score , precision_score , f1_score
import joblib
import numpy as np

def run_training_pipeline(data_path,model_path):
	df=load_data(data_path)

	X_train,y_train,X_val,y_val,X_test,y_test = split_data(df)

	pipeline=get_pipeline()
	pipeline.fit(X_train,y_train)

	y_probs = pipeline.predict_proba(X_val)[:,1]

	thresholds_prob = np.linspace(0,1,100)

	best_t = None
	best_f1=0
	for t in thresholds_prob:
		y_pred_t = (y_probs>=t).astype(int)
		recall = recall_score(y_val,y_pred_t)
		precision = precision_score(y_val,y_pred_t)
		f1 = f1_score(y_val,y_pred_t)
		
		if f1>best_f1:
			best_f1=f1
			best_t=t

		print(f"t : {t:.2f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

	print(f"Best threshold : {best_t} , Best F1 : {best_f1}")

	y_test_probs = pipeline.predict_proba(X_test)[:, 1]
	y_test_pred = (y_test_probs >= best_t).astype(int)
	print(classification_report(y_test,y_test_pred))

	precision , recall , thresholds = precision_recall_curve(y_test,y_test_probs)
	pr_auc = auc(recall,precision)
	print("PR-AUC: ", pr_auc) 

	print(f"ROC_AUC : ",roc_auc_score(y_test,y_test_probs))

	joblib.dump({"model":pipeline,"threshold":best_t}
		,f"{model_path}/model_pipeline.pkl")

