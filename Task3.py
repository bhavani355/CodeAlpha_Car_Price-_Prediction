# =========================================================
# CAR PRICE PREDICTION PROJECT
# FULLY CORRECTED WORKING CODE
# =========================================================

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    r"C:\Users\testm\OneDrive\Desktop\car data.csv"
)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

# =========================================================
# DISPLAY DATA
# =========================================================

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# =========================================================
# CONVERT CATEGORICAL DATA
# =========================================================

# Fuel Type
if 'Fuel_Type' in df.columns:
    df['Fuel_Type'] = df['Fuel_Type'].map({
        'Petrol': 0,
        'Diesel': 1,
        'CNG': 2
    })

# Seller Type
if 'Seller_Type' in df.columns:
    df['Seller_Type'] = df['Seller_Type'].map({
        'Dealer': 0,
        'Individual': 1
    })

# Transmission
if 'Transmission' in df.columns:
    df['Transmission'] = df['Transmission'].map({
        'Manual': 0,
        'Automatic': 1
    })

# =========================================================
# CREATE CAR AGE
# =========================================================

if 'Year' in df.columns:
    df['Car_Age'] = 2025 - df['Year']

# =========================================================
# CHECK AVAILABLE COLUMNS
# =========================================================

print("\n========== AVAILABLE COLUMNS ==========\n")
print(df.columns)

# =========================================================
# SELECT FEATURES
# =========================================================

features = []

possible_features = [
    'Present_Price',
    'Driven_kms',
    'Fuel_Type',
    'Seller_Type',
    'Transmission',
    'Owner',
    'Car_Age'
]

for col in possible_features:
    if col in df.columns:
        features.append(col)

print("\n========== FEATURES USED ==========\n")
print(features)

# =========================================================
# INPUT AND OUTPUT
# =========================================================

X = df[features]

y = df['Selling_Price']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MODEL TRAINING
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

score = r2_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========\n")
print("R2 Score :", round(score, 2))

# =========================================================
# GRAPH 1 : ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(10,6))

plt.scatter(y_test, y_pred)

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# GRAPH 2 : CAR AGE VS SELLING PRICE
# =========================================================

if 'Car_Age' in df.columns:

    plt.figure(figsize=(10,6))

    plt.scatter(df['Car_Age'], df['Selling_Price'])

    plt.title("Car Age vs Selling Price")
    plt.xlabel("Car Age")
    plt.ylabel("Selling Price")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

# =========================================================
# GRAPH 3 : SELLING PRICE DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,6))

plt.hist(df['Selling_Price'], bins=20)

plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# =========================================================
# SAMPLE PREDICTION
# =========================================================

print("\n========== SAMPLE PREDICTION ==========\n")

sample = X.iloc[0:1]

prediction = model.predict(sample)

print("Predicted Price :", round(prediction[0], 2))

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n========== PROJECT COMPLETED ==========")