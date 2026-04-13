### Amazon Alexa Customer Review Sentiment Analysis

**Project Overview**

This project focuses on analyzing Amazon Alexa customer reviews and classifying them as positive or negative using Natural Language Processing (NLP) and Machine Learning techniques. The model is built using a Random Forest Classifier trained on vectorized text data and product features.

**Objective**
- Analyze customer feedback data
- Perform text preprocessing and feature engineering
- Build a sentiment classification model
- Evaluate model performance using accuracy, confusion matrix and classification report

**Dataset**

The dataset used is the Amazon Alexa Reviews dataset, which contains:
- verified_reviews : customer review text
- date: review date
- variation : product type
- rating : star rating (Between 1 to 5)
- feedback → sentiment label (0 = negative, 1 = positive)

**Technologies Used:** 
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn (ML models & evaluation)
- CountVectorizer (NLP feature extraction)
- PyTorch
- Transformers

**Workflow**
1. Data Preprocessing: Handled missing values, Removed irrelevant columns, One-hot encoded categorical featur
2. Text Processing: Applied CountVectorizer to convert text into numerical features, Combined text features with structured data
3. Model Building: Split dataset into training and testing sets (80/20), Trained Random Forest Classifier, SVM and Logistic Regression
4. Model Evaluation: Confusion Matrix, Accuracy Score, Classification Report, ROC, Feature Importance
5. Use Advanced Model as LLM

**Results**
- SVM :94%
- Logistic Regression: 93%
- Random forest:93%

**Key Insights:**

- Model performs very well overall
- Strong performance on positive reviews
- Weak recall for negative reviews due to class imbalance

**How to Run the Project**

1. Clone the repository

   git clone https://github.com/parika8ec-hub/Assignment10.git
   
   cd Assignment10

2. Install dependencies
   
   pip install pandas numpy matplotlib seaborn scikit-learn
   
   pip install torch torchvision torchaudio

   pip install transformers
   
4. Run the notebook

Open Jupyter Notebook or Google Colab and run Assignment10.ipynb

**Future Improvements**

- Use TF-IDF instead of CountVectorizer
- Handle class imbalance (SMOTE or class weighting)
