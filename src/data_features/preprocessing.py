from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df):
    X=df.drop(columns=['Class'])
    y=df['Class']
    X_temp , X_test , y_temp , y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train , X_val , y_train , y_val = train_test_split(
        X_temp,y_temp,stratify=y_temp,random_state=42,test_size=0.25
    )
    return X_train,y_train,X_val,y_val,X_test,y_test