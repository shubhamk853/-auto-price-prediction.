# Auto Price Prediction
# Author: Shubham Bhagwan Kale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Dataset: UCI Automobile Dataset
# Download from: https://archive.ics.uci.edu/ml/datasets/automobile
# OR use this direct CSV link:
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/automobile.csv"

columns = [
    'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
    'num_doors', 'body_style', 'drive_wheels', 'engine_location',
    'wheel_base', 'length', 'width', 'height', 'curb_weight',
    'engine_type', 'num_cylinders', 'engine_size', 'fuel_system',
    'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm',
    'city_mpg', 'highway_mpg', 'price'
]

df = pd.read_csv(url, names=columns, na_values='?')
print("Dataset loaded! Shape:", df.shape)

# ============================================================
# 2. DATA CLEANING
# ============================================================
# Drop rows where price is missing (target variable)
df.dropna(subset=['price'], inplace=True)

# Fill missing numerical values with median
num_cols = ['normalized_losses', 'bore', 'stroke', 'horsepower', 'peak_rpm']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical with mode
df['num_doors'].fillna(df['num_doors'].mode()[0], inplace=True)

print("Missing values after cleaning:\n", df.isnull().sum().sum())

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n--- Basic Stats ---")
print(df[['engine_size', 'horsepower', 'price']].describe())

# Plot 1: Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], kde=True, color='steelblue')
plt.title('Car Price Distribution')
plt.xlabel('Price')
plt.savefig('price_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: price_distribution.png")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 7))
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, fmt='.1f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# Plot 3: Top 10 Makes by Average Price
plt.figure(figsize=(10, 5))
df.groupby('make')['price'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Car Makes by Average Price')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.savefig('top_makes_price.png', bbox_inches='tight')
plt.close()
print("Saved: top_makes_price.png")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# Label encode categorical columns
cat_cols = ['make', 'fuel_type', 'aspiration', 'num_doors', 'body_style',
            'drive_wheels', 'engine_location', 'engine_type',
            'num_cylinders', 'fuel_system']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# ============================================================
# 5. MODEL TRAINING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\n--- Model Results ---")
print(f"Linear Regression  -> R2: {lr_r2:.2f} | MAE: {lr_mae:.0f}")
print(f"Random Forest      -> R2: {rf_r2:.2f} | MAE: {rf_mae:.0f}")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
feat_imp.sort_values(ascending=False).head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ============================================================
# 7. ACTUAL vs PREDICTED PLOT
# ============================================================
plt.figure(figsize=(7, 5))
plt.scatter(y_test, rf_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Price (R² = {rf_r2:.2f})')
plt.savefig('actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("Saved: actual_vs_predicted.png")

print("\nDone! Best Model: Random Forest with R2 =", round(rf_r2, 2))
