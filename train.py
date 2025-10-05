# src/train.py
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import misc

def main():
    print("=== DecisionTreeRegressor Training Started ===")

    # Load and preprocess data
    df = misc.load_data()
    X, y = misc.preprocess(df)

    # Split data into train/test
    X_train, X_test, y_train, y_test = misc.split_data(X, y)

    # Scale features (optional but keeps pipeline consistent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define model
    model = DecisionTreeRegressor(random_state=42)

    # Train model
    model = misc.train_model(model, X_train_scaled, y_train)

    # Evaluate on test set
    regression_mse = misc.evaluate_model(model, X_test_scaled, y_test)

    print(f"MODEL: DecisionTreeRegressor | Regression_MSE: {regression_mse:.4f}")
    print("=== Training Completed ===")

if __name__ == "__main__":
    main()