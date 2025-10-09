# train2.py
import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import misc

def main():
    print("=== KernelRidge Training Started ===")

    # Load and preprocess data
    df = misc.load_data()
    X, y = misc.preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = misc.split_data(X, y)

    # Scale features (important for KernelRidge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define model (RBF kernel)
    model = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1)

    # Train model
    model = misc.train_model(model, X_train_scaled, y_train)

    # Evaluate on test set
    Kernel_mse = misc.evaluate_model(model, X_test_scaled, y_test)

    print(f"MODEL: KernelRidge | Kernel_MSE: {Kernel_mse:.4f}")
    print("=== Training Completed ===")

if __name__ == "__main__":
    main()