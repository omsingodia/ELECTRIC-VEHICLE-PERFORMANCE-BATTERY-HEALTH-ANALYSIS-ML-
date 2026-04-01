import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import statsmodels.api as sm

df = pd.read_csv("/content/electric_vehicle_analytics CHANGED.csv")

df.drop(columns=["Vehicle_ID"], errors="ignore", inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Battery_Health_%", axis=1)
y = df["Battery_Health_%"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

model_map = {}

# LR
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
results.append(["Linear Regression", r2_score(y_test, pred)])
model_map["Linear Regression"] = lr

# KNN
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
results.append(["KNN", r2_score(y_test, pred)])
model_map["KNN"] = knn

# SVM
svm = SVR()
svm.fit(X_train_scaled, y_train)
pred = svm.predict(X_test_scaled)
results.append(["SVM", r2_score(y_test, pred)])
model_map["SVM"] = svm

# RF (BEST)
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
results.append(["Random Forest", r2_score(y_test, pred)])
model_map["Random Forest"] = rf

results_df = pd.DataFrame(results, columns=["Model", "R2 Score"])
print("\n===== MODEL COMPARISON =====\n")
print(results_df)

best = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
best_model = model_map[best["Model"]]

print("\nBEST MODEL:", best["Model"])

print("\n===== ENTER DETAILS (y = enter, n = skip) =====\n")

def take_input(name):
    ch = input(f"{name}? (y/n): ").lower()
    if ch == 'y':
        return float(input(f"Enter {name}: "))
    return 0

input_data = {}

input_data["Year"] = take_input("Year")
input_data["Battery_Capacity_kWh"] = take_input("Battery Capacity")
input_data["Range_km"] = take_input("Range")
input_data["Charging_Power_kW"] = take_input("Charging Power")
input_data["Charging_Time_hr"] = take_input("Charging Time")
input_data["Charge_Cycles"] = take_input("Charge Cycles")
input_data["Energy_Consumption_kWh_per_100km"] = take_input("Energy Consumption")
input_data["Mileage_km"] = take_input("Mileage")
input_data["Avg_Speed_kmh"] = take_input("Speed")
input_data["Temperature_C"] = take_input("Temperature")

print("\nChoose Brand (type exact name or n to skip):")
print("Tesla, BMW, Nissan, Kia, Hyundai, Ford, Chevrolet, Volkswagen")

brand = input("Enter Brand: ").lower()

brands = {
    "bmw": "Make_BMW",
    "chevrolet": "Make_Chevrolet",
    "ford": "Make_Ford",
    "hyundai": "Make_Hyundai",
    "kia": "Make_Kia",
    "nissan": "Make_Nissan",
    "tesla": "Make_Tesla",
    "volkswagen": "Make_Volkswagen"
}

for key in brands.values():
    input_data[key] = 0

if brand in brands:
    input_data[brands[brand]] = 1

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

if best["Model"] in ["KNN", "SVM"]:
    input_df = scaler.transform(input_df)

prediction = best_model.predict(input_df)

print("\n===== RESULT =====")
print(f"Predicted Battery Health: {prediction[0]:.2f}%")
