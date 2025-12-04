import pandas as pd
import numpy as np

def train_and_save_model(train_csv_path="mnist_train.csv", model_path="gnb_model.npz"):
    """
    Loads training data, calculates Gaussian Naive Bayes parameters (priors, means,
    variances), and saves them to a compressed NumPy file.
    """
    print("Loading training data...")
    # Load data
    train_df = pd.read_csv(train_csv_path)

    # Separate features (X) from labels (y)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]

    # Get classes and feature count
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    
    print(f"Data loaded: {len(y_train)} samples, {n_features} features, {n_classes} classes.")
    print("Calculating model parameters (priors, means, variances)...")

    # --- 1. Calculate Priors (P(Ck)) ---
    # value_counts() gives counts, sort_index() ensures 0-9 order
    # .values gets the NumPy array, / len(y_train) makes it a probability
    priors = (y_train.value_counts().sort_index() / len(y_train)).values
    
    # --- 2. Calculate Means & Variances (μ_ki, σ²_ki) ---
    # We group the feature data (X_train) by the labels (y_train)
    grouped_by_class = X_train.groupby(y_train)
    
    # .mean() and .var() compute the stats for each column (pixel)
    # This results in a (10, 784) DataFrame for each
    means_df = grouped_by_class.mean()
    vars_df = grouped_by_class.var()

    # --- 3. Handle Zero Variance (Smoothing) ---
    # If a pixel is always off (0) for a digit, variance is 0
    # This causes division-by-zero in the Gaussian formula.
    # We add a small "epsilon" to all variances for numerical stability.
    epsilon = 1e-9
    vars_df += epsilon
    
    # Convert from Pandas DataFrames to NumPy arrays for saving
    means = means_df.values
    variances = vars_df.values
    
    # --- 4. Save Model Parameters ---
    # We save the calculated arrays to a single compressed file.
    np.savez_compressed(
        model_path,
        priors=priors,
        means=means,
        variances=variances,
        classes=classes
    )
    
    print(f"Model parameters successfully trained and saved to '{model_path}'.")

if __name__ == "__main__":
    train_and_save_model()