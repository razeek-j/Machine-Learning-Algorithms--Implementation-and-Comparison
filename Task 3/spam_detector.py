import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_text(text):
    """
    Cleans text data by:
    1. Removing punctuation
    2. Converting to lowercase
    3. Removing numbers (less important for spam, but good practice)
    """
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) # Remove punctuation and numbers
    text = text.strip() # Remove leading/trailing whitespace
    return text

def load_data(filepath="SMSSpamCollection"):
    """
    Loads the TSV data, gives it column names, and cleans the text.
    """
    print(f"Loading data from '{filepath}'...")
    # The file is TAB-separated, not comma-separated
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    
    # Clean the messages
    print("Cleaning text data...")
    df['message'] = df['message'].apply(clean_text)
    
    # Map labels to numbers (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Data loaded: {len(df)} messages ({df['label'].sum()} spam, {len(df) - df['label'].sum()} ham)")
    return df['message'], df['label']

def run_experiments():
    """
    Loads data, splits it, and runs both n-gram and RF models.
    """
    X, y = load_data()
    
    # Split the data into 80% training and 20% testing
    # random_state=42 ensures we get the same split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")
    
    # Define the class names for reports
    class_names = ['ham', 'spam']

    # ---
    # Experiment 1: N-Gram Model (using Multinomial Naive Bayes)
    # ---
    print("\n" + "="*60)
    print("--- STARTING: Experiment 1: N-Gram + Naive Bayes ---")
    print("="*60)
    
    # We create a "Pipeline" to chain the steps together.
    # 1. 'vect': Use CountVectorizer (creates n-grams)
    # 2. 'clf': Use MultinomialNB (the classifier)
    ngram_model = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))), # Use 1-grams and 2-grams
        ('clf', MultinomialNB())
    ])
    
    # Train the n-gram model
    print("Training N-Gram model...")
    start_time = time.time()
    ngram_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.4f} seconds.")
    
    # Test the n-gram model
    print("Evaluating N-Gram model...")
    y_pred_ngram = ngram_model.predict(X_test)
    
    # Print N-Gram results
    print("\n--- N-Gram Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_ngram) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_ngram))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_ngram, target_names=class_names))

    # ---
    # Experiment 2: Random Forest Model (using TF-IDF)
    # ---
    print("\n" + "="*60)
    print("--- STARTING: Experiment 2: TF-IDF + Random Forest ---")
    print("="*60)
    
    # Create the pipeline for the Random Forest
    # 1. 'tfidf': Use TfidfVectorizer (converts text to numerical matrix)
    # 2. 'clf': Use RandomForestClassifier
    rf_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])
    
    # Train the RF model
    print("Training Random Forest model...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.4f} seconds.")
    
    # Test the RF model
    print("Evaluating Random Forest model...")
    y_pred_rf = rf_model.predict(X_test)
    
    # Print RF results
    print("\n--- Random Forest Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

if __name__ == "__main__":
    run_experiments()