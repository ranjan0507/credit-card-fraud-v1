from src.pipelines.training import run_training_pipeline

if __name__=="__main__":
	run_training_pipeline(data_path="data/creditcard.csv",model_path="artifacts")
