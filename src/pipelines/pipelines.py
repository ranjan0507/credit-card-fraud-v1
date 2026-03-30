from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def get_pipeline():
	preprocessor = ColumnTransformer(transformers=[
		("scaling",StandardScaler(),["Amount"])
	],remainder="passthrough")

	pipeline = Pipeline(
		steps=[
			('preprocessing',preprocessor),
			('smote',SMOTE(random_state=42)),
			('training',LogisticRegression(max_iter=2000,random_state=42))
		]
	)

	return pipeline