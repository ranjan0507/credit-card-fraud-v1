import pandas as pd
import os

def load_data(path='data/creditcard.csv'):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    resolved_path = path if os.path.isabs(path) else os.path.join(project_root, path)

    print('Loading data')
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Data file not found: {resolved_path}")

    df = pd.read_csv(resolved_path)
    print('Data loaded')
    return df

def describe_data(df:pd.DataFrame):
    print(f'Shape         : {df.shape}')                       
    print(f'Fraud cases   : {df["Class"].sum()}')           
    print(f'Legit cases   : {(df["Class"]==0).sum()}')        
    fraud_pct = df['Class'].mean() * 100
    print(f'Fraud rate    : {fraud_pct:.4f}%')                 
    print(f'Amount range  : ${df["Amount"].min():.2f} to ${df["Amount"].max():.2f}')
