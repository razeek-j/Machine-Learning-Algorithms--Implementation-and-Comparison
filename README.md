# Machine Learning Algorithms: Implementation & Comparison

## Overview

This repository serves as a comprehensive collection of fundamental Machine Learning implementations, focusing on both computer vision and natural language processing (NLP) tasks. It provides a professional comparison of different algorithmic approaches—ranging from probabilistic models built from scratch to robust ensemble methods using modern libraries.

The projects demonstrate a deep understanding of core ML concepts, including **Gaussian Naive Bayes**, **Random Forests**, **N-Gram models**, and **TF-IDF vectorization**.

## Projects

### 1. Handwritten Digit Classification (Gaussian Naive Bayes)
**Focus:** Probabilistic Modeling & Custom Implementation

This module implements a **Gaussian Naive Bayes** classifier from the ground up to recognize handwritten digits from the MNIST dataset. Instead of relying solely on high-level APIs, this implementation explicitly calculates the underlying statistical parameters.

*   **Key Techniques:**
    *   **Manual Parameter Estimation:** Explicit calculation of class priors ($P(C_k)$), feature means ($\mu_{ki}$), and variances ($\sigma^2_{ki}$).
    *   **Numerical Stability:** Implementation of epsilon smoothing to handle zero-variance cases and prevent numerical instability.
    *   **Vectorized Operations:** Efficient data processing using NumPy and Pandas.

### 2. Handwritten Digit Classification (Random Forest)
**Focus:** Ensemble Learning & Scalability

Building upon the digit classification challenge, this module leverages the **Random Forest** algorithm to achieve higher accuracy and robustness. It demonstrates the power of ensemble methods in handling high-dimensional image data.

*   **Key Techniques:**
    *   **Ensemble Learning:** Aggregating predictions from 100 decision trees to reduce overfitting and variance.
    *   **Parallel Processing:** Utilizing multi-core processing (`n_jobs=-1`) for efficient model training.
    *   **Model Persistence:** efficient serialization of trained models using `joblib` for deployment readiness.

### 3. SMS Spam Detection (NLP Comparison)
**Focus:** Natural Language Processing & Model Benchmarking

This module addresses a classic binary classification problem in NLP: distinguishing between legitimate (ham) and spam SMS messages. It conducts a comparative analysis between two distinct approaches to text classification.

*   **Approaches Compared:**
    *   **N-Gram Model:** Utilizes a **Bag-of-Words** approach with **Bigrams** (1-gram + 2-gram) fed into a **Multinomial Naive Bayes** classifier. This captures local word context effectively.
    *   **TF-IDF + Random Forest:** Employs **Term Frequency-Inverse Document Frequency** to weigh word importance, coupled with a Random Forest classifier for robust decision boundaries.
*   **Key Techniques:**
    *   **Text Preprocessing:** Regex-based cleaning, lowercasing, and tokenization.
    *   **Scikit-learn Pipelines:** Streamlined workflows combining vectorization and classification.
    *   **Performance Metrics:** Detailed evaluation using Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).

## Technical Stack

*   **Languages:** Python 3.x
*   **Core Libraries:**
    *   **NumPy & Pandas:** Data manipulation and vectorization.
    *   **Scikit-learn:** Model building, evaluation, and preprocessing pipelines.
    *   **Joblib:** Model serialization.

## Getting Started

Follow these steps to set up the project locally on your machine.

### Prerequisites

*   **Python 3.8+** installed.
*   **Git** installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set up a Virtual Environment (Recommended):**
    It is best practice to use a virtual environment to manage dependencies.
    ```bash
    # Create virtual environment
    python3 -m venv venv

    # Activate virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages for all tasks.
    ```bash
    pip install pandas numpy scikit-learn joblib
    # Or install from specific requirements files if preferred:
    # pip install -r "Task 1/requirements.txt"
    ```

### Data Setup

Since large datasets are not stored in the repository, you need to download them:

*   **Task 1 & 2 (MNIST):**
    1.  Download `mnist_train.csv` and `mnist_test.csv` from [pjreddie.com](https://pjreddie.com/projects/mnist-in-csv/).
    2.  Place them in both `Task 1/` and `Task 2/` directories.

*   **Task 3 (SMS Spam):**
    1.  Download the SMS Spam Collection from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
    2.  Extract the zip file.
    3.  Place the `SMSSpamCollection` file in the `Task 3/` directory.

### Running the Projects

**Task 1: Gaussian Naive Bayes**
```bash
cd "Task 1"
python train.py
# This will train the model and save 'gnb_model.npz'
```

**Task 2: Random Forest**
```bash
cd "../Task 2"
python train.py
# This will train the model and save 'rf_model.joblib'
```

**Task 3: SMS Spam Detection**
```bash
cd "../Task 3"
python spam_detector.py
# This will run the comparison between N-Gram and Random Forest models
```

---
*This repository is maintained for educational and professional demonstration purposes, showcasing the practical application of machine learning theory.*
