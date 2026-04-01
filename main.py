import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("/content/electric_vehicle_analytics CHANGED.csv")

df.drop(columns=["Vehicle_ID"], errors="ignore", inplace=True)

# save original columns
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Battery_Health_%", axis=1)
y = df["Battery_Health_%"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=300,max_depth=10,random_state=42)
}

results = []

for name, model in models.items():

    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results.append([name, r2, mae, rmse])


results_df = pd.DataFrame(results, columns=["Model", "R2", "MAE", "RMSE"])

print("\n===== MODEL COMPARISON MATRIX =====\n")
print(results_df)

best = results_df.sort_values(by="R2", ascending=False).iloc[0]
best_model_name = best["Model"]
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name)


print("\n===== ENTER INPUT (y = give value, n = skip → 0) =====")

input_data = {}

for col in X.columns:
    choice = input(f"{col}? (y/n): ").lower()

    if choice == "y":
        val = float(input(f"Enter value for {col}: "))
        input_data[col] = val
    else:
        input_data[col] = 0


input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

if best_model_name in ["KNN", "SVM"]:
    input_df = scaler.transform(input_df)

prediction = best_model.predict(input_df)

print("\n===== RESULT =====")
print(f"Predicted Battery Health: {prediction[0]:.2f}%")
