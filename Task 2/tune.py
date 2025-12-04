import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import time
import os

def run_experiment():
    """
    Loads MNIST data once, then trains and evaluates several different
    Random Forest models to compare their hyperparameters.
    """
    
    # --- 1. Load Data (Once) ---
    print("Loading training and test data...")
    train_df = pd.read_csv("mnist_train.csv", header=None)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:].values 

    test_df = pd.read_csv("mnist_test.csv", header=None)
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:].values
    
    print("Data loaded.")

    # --- 2. Define Models to Test ---
    # We create a dictionary of models to try.
    # All models get n_jobs=-1 (use all cores) and random_state=42 (reproducible)
    
    models_to_test = {
        "baseline_100_trees_unlimited_depth": RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        ),
        "more_trees_200_unlimited_depth": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42
        ),
        "limited_depth_100_trees_max_depth_20": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Get class names for the report
    class_names = [str(c) for c in train_df.iloc[:, 0].unique()]
    class_names.sort()

    # --- 3. Loop, Train, and Evaluate ---
    for model_name, model in models_to_test.items():
        print("\n" + "="*60)
        print(f"--- STARTING: {model_name} ---")
        print("="*60)
        
        # --- Train ---
        print("Training model...")
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        train_time = end_train - start_train
        print(f"Training completed in {train_time:.2f} seconds.")

        # --- Save Model (Optional) ---
        model_filename = f"{model_name}.joblib"
        joblib.dump(model, model_filename)
        model_size = os.path.getsize(model_filename) / (1024*1024) # in MB
        print(f"Model saved to '{model_filename}' (Size: {model_size:.2f} MB)")
        
        # --- Predict ---
        print("Making predictions on test set...")
        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        pred_time = end_pred - start_pred
        print(f"Predictions made in {pred_time:.2f} seconds.")

        # --- Report ---
        print("\n--- Evaluation Results ---")
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)

    print("\n" + "="*60)
    print("--- ALL EXPERIMENTS COMPLETE ---")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
