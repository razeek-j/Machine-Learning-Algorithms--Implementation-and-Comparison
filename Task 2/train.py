import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

def train_and_save_model(train_csv_path="mnist_train.csv", model_path="rf_model.joblib"):
    """
    Loads training data, trains a Random Forest classifier, and saves the
    trained model to a file using joblib.
    """
    
    # --- 1. Load Data ---
    print("Loading training data...")
    start_load = time.time()
    # Tell pandas there is NO header row in the CSV file
    train_df = pd.read_csv(train_csv_path, header=None)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:] # Correctly select all pixel columns
    end_load = time.time()
    print(f"Data loaded in {end_load - start_load:.2f} seconds.")
    print(f"Beginning Random Forest training...")
    print("This will take several minutes...")

    # --- 2. Initialize Model ---
    # We use the RandomForestClassifier from scikit-learn
    # n_estimators=100 means we will build 100 separate decision trees.
    # n_jobs=-1 tells the model to use ALL available CPU cores to speed up training.
    # random_state=42 ensures you get the same results every time you run it.
    model = RandomForestClassifier(
        n_estimators=100, 
        n_jobs=-1, 
        random_state=42
    )

    # --- 3. Train the Model ---
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    print(f"Model training completed in {(end_train - start_train) / 60:.2f} minutes.")

    # --- 4. Save the Model ---
    # We use joblib.dump to save the trained model object.
    # This is more efficient than .npz for complex sklearn models.
    print(f"Saving model to '{model_path}'...")
    joblib.dump(model, model_path)
    
    print("Model successfully trained and saved.")

if __name__ == "__main__":
    train_and_save_model()

