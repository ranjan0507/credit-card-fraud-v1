from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df):
    return train_test_split(
        df, df['Class'], test_size=0.2, stratify=df['Class'], random_state=42
    )