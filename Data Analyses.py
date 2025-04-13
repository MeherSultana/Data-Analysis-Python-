
#**first we will import all the libraries we need**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# We will just read the head so we will have the idea what kind of data we have, althought you always read the data carefully

data = pd.read_csv("/content/Datacsv")
print(data.head())

#To get more info
data.info()

#If there are any duplicate values we can drop it
data.drop_duplicates

#1. Descriptive Analysis
data.describe()

#Distribution of categorical variables
data['Gender'].value_counts()

#Correlation between numerical variables (e.g., Tenure vs. TotalCharges).
data[['Tenure', 'TotalCharges']].corr()

#Analysis of churn behavior by key factors (e.g., monthly charges, tenure).
data.groupby('Churn')[['MonthlyCharges', 'Tenure']].mean()

#Build a logistic regression or decision tree model to predict churn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# 1. Data Preprocessing
# Handle missing values (fill NaN in InternetService with 'Unknown')
data['InternetService'] = data['InternetService'].fillna('Unknown')

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and Target
X = data.drop(['CustomerID', 'Churn'], axis=1)  # Drop irrelevant columns and target
y = data['Churn']

# Standardize numerical features
scaler = StandardScaler()
X[['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    X[['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']]
)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Model")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# 3. Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Model")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Optional: Feature Importance for Decision Tree
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_})
print("\nFeature Importances (Decision Tree):")
print(feature_importances.sort_values(by='Importance', ascending=False))
