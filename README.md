# 📧 Email & SMS Spam Classifier

A machine learning based web application that classifies email/SMS messages as Spam or Ham.

## 🚀 Live Demo

Coming soon...

## 🧠 Machine Learning

The project uses:

- Text preprocessing
- Tokenization
- Stopword removal
- Stemming
- TF-IDF Vectorization
- Multinomial Naive Bayes

## 🔄 Workflow

User Message
↓
Text Preprocessing
↓
TF-IDF
↓
Multinomial Naive Bayes
↓
Spam / Ham

## 📊 Model Selection

Multiple machine learning algorithms were experimented with during model development.

The final deployed model is Multinomial Naive Bayes using TF-IDF features.

Model selection was based on evaluation metrics including accuracy and precision.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib

## 💻 Run Locally

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd email-sms-spam-classifier