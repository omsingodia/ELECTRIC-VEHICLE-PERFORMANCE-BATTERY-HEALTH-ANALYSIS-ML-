import pandas as pd

def load_data():
    path = r"D:\SEM. 4\ML\project\electric_vehicle_analytics CHANGED.csv"
    df = pd.read_csv(path)
    return df
