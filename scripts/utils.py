import pandas as pd

def load_dataset():
    monthly_data = pd.read_csv("../dataset/housing_in_london_monthly_variables.csv")
    monthly_data["date"] = pd.to_datetime(monthly_data["date"])
    yearly_data = pd.read_csv("../dataset/housing_in_london_yearly_variables.csv")
    yearly_data["date"] = pd.to_datetime(yearly_data["date"])
    return monthly_data, yearly_data

def get_area(data, area):
    return data[data["area"] == area]