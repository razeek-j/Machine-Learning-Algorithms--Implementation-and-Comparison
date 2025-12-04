import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model(model_path="gnb_model.npz"):
    """Loads the pre-calculated model parameters from a .npz file."""
    data = np.load(model_path)
    return data['priors'], data['means'], data['variances'], data['classes']

def predict(X, priors, means, variances, classes):
    """
    Predicts class labels for samples in X using GNB parameters.
    
    X can be a Pandas DataFrame or a NumPy array.
    """
    n_samples = X.shape[0]
    n_classes = len(classes)
    
    # Convert X to NumPy array if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    # This will store the log-posterior probabilities for each class
    log_posteriors = np.zeros((n_samples, n_classes))
    
    # Calculate log-posterior for each class k
    for k_idx, k_class in enumerate(classes):
        
        # Get parameters for this specific class
        prior_k = priors[k_idx]
        means_k = means[k_idx, :]   # (1, 784) vector
        vars_k = variances[k_idx, :] # (1, 784) vector

        # --- Calculate Log-Likelihood: log(P(x | Ck)) ---
        # We use the log-probability formula to avoid numerical underflow
        
        # 1. First term: -0.5 * sum(log(2*pi*sigma_k^2))
        # This is constant for all samples, so we calculate it once
        term1 = -0.5 * np.sum(np.log(2. * np.pi * vars_k))

        # 2. Second term: -0.5 * sum( (x_i - mu_k)^2 / sigma_k^2 )
        # This is calculated for every sample.
        # (X - means_k) is (n_samples, 784)
        # (vars_k) is (1, 784)
        # We sum across axis=1 (the 784 features)
        term2 = -0.5 * np.sum(((X - means_k) ** 2) / vars_k, axis=1)

        # Log-Posterior = log(Prior) + Log-Likelihood
        # log(P(Ck | x)) ~ log(P(Ck)) + log(P(x | Ck))
        # log(P(x | Ck)) = term1 + term2
        log_posteriors[:, k_idx] = np.log(prior_k) + term1 + term2

    # The prediction is the class with the highest log-posterior probability
    return np.argmax(log_posteriors, axis=1)

if __name__ == "__main__":
    print("Loading trained model parameters...")
    # 1. Load the trained model
    priors, means, variances, classes = load_model("gnb_model.npz")

    print("Loading test data...")
    # 2. Load the test data
    test_df = pd.read_csv("mnist_test.csv")
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    
    print("Making predictions on test set...")
    # 3. Make predictions
    y_pred = predict(X_test, priors, means, variances, classes)
    
    print("\n--- Evaluation Results ---")
    
    # 4. Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Calculate and print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 6. Print Precision, Recall, F1-Score
    print("\nClassification Report:")
    # Convert class labels (0, 1, 2...) to strings ("0", "1", "2"...)
    class_names = [str(c) for c in classes]
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)