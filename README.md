# README: Sentiment Analysis with Logistic Regression and TF-IDF

---

### **Project Overview**

This project implements a sentiment analysis model using Logistic Regression and TF-IDF Vectorizer. The dataset consists of book reviews, which are classified into positive or negative sentiments. The tasks include feature engineering, model training, evaluation, and optimization using a pipeline and hyperparameter tuning.

---

### **Features**

- **Text Preprocessing**: Utilizes `TF-IDF Vectorizer` to transform text data into numerical representations.
- **Model Training**: Trains a Logistic Regression model for binary classification.
- **Performance Evaluation**: Evaluates the model using accuracy, ROC-AUC, and confusion matrix.
- **Optimization**: Applies hyperparameter tuning with `GridSearchCV` to enhance model performance.

---

### **Key Tasks**

1. **Load the Dataset**:
   - File: `data/bookReviewsData.csv`
   - The dataset includes two columns:
     - `Review`: Text data (features).
     - `Positive Review`: Binary labels indicating sentiment.

2. **Exploratory Data Analysis**:
   - Extracts features (`Review`) and labels (`Positive Review`).
   - Splits the dataset into training and test sets (67/33 split).

3. **Text Vectorization**:
   - Converts text to numerical data using `TF-IDF Vectorizer` with bi-gram strategies and minimum document frequency.

4. **Model Training**:
   - Trains a Logistic Regression model with 200 iterations for convergence.
   - Predicts probabilities and class labels for the test data.

5. **Model Evaluation**:
   - Computes metrics including:
     - **Accuracy**: 81.9%
     - **AUC-ROC**: 91.0%
   - Visualizes ROC curve to evaluate classification performance.

6. **Model Optimization**:
   - Builds a pipeline integrating vectorization and classification.
   - Performs hyperparameter tuning using `GridSearchCV` with:
     - Regularization strength (`C`): [0.1, 1, 10].
     - N-gram range: [(1, 1), (1, 2)].
   - Identifies optimal parameters for improved performance.

---

### **How to Run the Project**

1. **Setup Environment**:
   - Install required libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn
     ```
   - Ensure `bookReviewsData.csv` is placed in the `data` folder.

2. **Run the Python Script or Notebook**:
   - Load the dataset.
   - Execute the code sequentially to preprocess data, train the model, and evaluate performance.

3. **Hyperparameter Tuning**:
   - Use `GridSearchCV` to optimize model parameters.
   - Visualize the final ROC curve and report the best parameters.

---

### **Performance Metrics**

- **Accuracy**: 81.9%
- **AUC-ROC**: 91.0%
- **Feature Space**: ~19,000 features (based on vocabulary size).

---

### **Findings**

1. **Model Performance**:
   - The Logistic Regression model with TF-IDF transformation performed well, achieving high accuracy and AUC-ROC scores.
   - Bi-gram strategies and feature space optimization improved text representation.

2. **Optimization Results**:
   - Grid Search identified optimal hyperparameters, significantly enhancing the model's predictive power.

3. **Insights**:
   - TF-IDF and Logistic Regression effectively classify textual data in binary sentiment analysis tasks.
   - Hyperparameter tuning and bi-gram strategies contribute to improved generalization.

---

### **Future Improvements**

- Experiment with advanced classification models (e.g., Random Forest, SVM, or neural networks).
- Use larger datasets for better generalization.
- Incorporate additional text preprocessing techniques like stemming or lemmatization.

---

### **Dependencies**

- Python 3.7+
- Required Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

