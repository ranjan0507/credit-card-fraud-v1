from src.data.data_loader import load_data
from src.data.preprocessing import split_data
from src.pipelines.pipelines import get_pipeline
from sklearn.metrics import classification_report , roc_auc_score
import joblib


def run_training_pipeline(data_path,model_path):
	df=load_data(data_path)

	X_train,X_test,y_train,y_test = split_data(df)

	pipeline=get_pipeline()
	pipeline.fit(X_train,y_train)

	y_pred=pipeline.predict(X_test)

	print(classification_report(y_test,y_pred))
	print("ROC-AUC score : " , roc_auc_score(y_test,y_pred))

	joblib.dump(pipeline,f"{model_path}/model_pipeline.pkl")

