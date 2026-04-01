import pandas as pd
import numpy as np

df = pd.read_csv("electric_vehicle_analytics CHANGED (1).csv")

df.head()
df = df.drop(["Make", "Model", "Vehicle_Type"], axis=1)

y_reg = df["Battery_Health_%"]

df["Battery_Status"] = df["Battery_Health_%"].apply(lambda x: 1 if x > 70 else 0)
y_clf = df["Battery_Status"]

X = df.drop(["Battery_Health_%", "Battery_Status"], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train_reg)

y_pred_lr = lr.predict(X_test)

print("Linear Regression")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_lr))
print("R2:", r2_score(y_test_reg, y_pred_lr))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train_reg)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_rf))
print("R2:", r2_score(y_test_reg, y_pred_rf))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_clf)

y_pred_log = log_reg.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print(classification_report(y_test_clf, y_pred_log))
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train_clf)

y_pred_svm = svc.predict(X_test)

print("\nSVM")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_svm))
print(classification_report(y_test_clf, y_pred_svm))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train_clf)

y_pred_knn = knn.predict(X_test)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_knn))
print(classification_report(y_test_clf, y_pred_knn))
print("\n--- Regression Comparison ---")
print("Linear R2:", r2_score(y_test_reg, y_pred_lr))
print("Random Forest R2:", r2_score(y_test_reg, y_pred_rf))
print("\n--- Classification Comparison ---")
print("Logistic:", accuracy_score(y_test_clf, y_pred_log))
print("SVM:", accuracy_score(y_test_clf, y_pred_svm))
print("KNN:", accuracy_score(y_test_clf, y_pred_knn))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_clf, y_pred_svm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (SVM)")
plt.show()
