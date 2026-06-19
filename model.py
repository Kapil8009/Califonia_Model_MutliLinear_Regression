import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_save_model():
    # 1. Load the California Housing dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {housing.feature_names}")
    print(f"Target: Price (in $100,000)")
    
    # 2. Exploratory Data Analysis
    print("\n--- Dataset Information ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # 3. Feature Selection (all features for multi-linear regression)
    X = df[housing.feature_names]
    y = df['Price']
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Multi-Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 7. Model Evaluation
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    print("\n--- Model Performance ---")
    print(f"Training R² Score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Testing R² Score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    
    # 8. Feature Importance (Coefficients)
    feature_importance = pd.DataFrame({
        'Feature': housing.feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    print("\n--- Feature Coefficients ---")
    print(feature_importance)
    
    # 9. Visualizations
    # Create plots directory
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('static/plots/actual_vs_predicted.png')
    plt.close()
    
    # Residual Plot
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('static/plots/residuals.png')
    plt.close()
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    feature_importance.plot(x='Feature', y='Coefficient', kind='bar', legend=False)
    plt.title('Feature Importance (Coefficients)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('static/plots/correlation_heatmap.png')
    plt.close()
    
    # 10. Save Model and Scaler
    if not os.path.exists('model'):
        os.makedirs('model')
    
    joblib.dump(model, 'model/housing_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    
    print("\nModel and scaler saved successfully!")
    
    return model, scaler, feature_importance

if __name__ == "__main__":
    train_and_save_model()