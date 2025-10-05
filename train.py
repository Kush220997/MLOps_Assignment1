# src/train.py
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
import misc

def main():
    print("=== DecisionTreeRegressor Training Started ===")

    # Load and preprocess data
    df = misc.load_data()
    X, y = misc.preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = misc.split_data(X, y)

    # Define model
    model = DecisionTreeRegressor(random_state=42)

    # Train model
    model = misc.train_model(model, X_train, y_train)

    # Evaluate on test set
    regression_mse = misc.evaluate_model(model, X_test, y_test)

    print(f"MODEL: DecisionTreeRegressor | MSE: {regression_mse:.4f}")
    print("=== Training Completed ===")

if __name__ == "__main__":
    main()
