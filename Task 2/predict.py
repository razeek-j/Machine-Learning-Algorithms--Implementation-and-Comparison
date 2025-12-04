import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

def load_model(model_path="rf_model.joblib"):
    """Loads the pre-trained Random Forest model from a .joblib file."""
    print(f"Loading model from '{model_path}'...")
    return joblib.load(model_path)

if __name__ == "__main__":
    # 1. Load the trained model
    model = load_model("rf_model.joblib")

    # 2. Load the test data
    print("Loading test data...")
    test_df = pd.read_csv("mnist_test.csv")
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:] # Correctly select all pixel columns
    
    print("Making predictions on test set...")
    # 3. Make predictions
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()
    print(f"Predictions made in {end_pred - start_pred:.2f} seconds.")

    
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
    # Get the class names from the model's learned classes
    class_names = [str(c) for c in model.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
