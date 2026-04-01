import pandas as pd
def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["Vehicle_ID"], errors="ignore", inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df
def split_features(df):
    X = df.drop("Battery_Health_%", axis=1)
    y = df["Battery_Health_%"]
    return X, y
