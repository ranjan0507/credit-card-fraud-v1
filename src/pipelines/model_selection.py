from src.data_features.data_loader import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from src.pipelines.pipelines import get_pipeline
from sklearn.metrics import average_precision_score
from src.data_features.preprocessing import split_data
import numpy as np

def get_models():
	return {
		"log_reg":LogisticRegression(
			solver="liblinear",
			class_weight="balanced",
			max_iter=2000,
			random_state=42
		),
		"rand_forest":RandomForestClassifier(
			n_estimators=100,
			n_jobs=-1,
			random_state=42
		)
	}

def evaluate_model_cv(X,y,model):
	skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
	scores = []

	for fold , (train_idx , val_idx) in enumerate(skf.split(X,y)):

		X_train,X_val = X.iloc[train_idx] , X.iloc[val_idx]
		y_train,y_val = y.iloc[train_idx] , y.iloc[val_idx]

		pipeline = get_pipeline(model)
		pipeline.fit(X_train,y_train)

		y_probs = pipeline.predict_proba(X_val)[:,1]
		pr_auc = average_precision_score(y_val,y_probs)
		print(f"FOLD : {fold}  PR-AUC : {pr_auc}")

		scores.append(pr_auc)

	return scores

def run_model_selection(path):
	df = load_data(path)
	X=df.drop('Class',axis=1)
	y=df['Class']
	
	models = get_models()
	results = {}

	for name , model in models.items():
		print(f"Evaluating {model}")
		scores = evaluate_model_cv(X,y,model)
		print(f"Mean pr-auc : {np.mean(scores):.4f}")
		print(f"Std pr-auc : {np.std(scores):.4f}")
		results[name]=np.mean(scores)
	
	best_model_name = max(results,key=results.get)

	print(f"Best Model : {best_model_name}")
	return best_model_name

if __name__=="__main__":
	best = run_model_selection("data/raw/creditcard.csv")
