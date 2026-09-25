import pandas as pd
import nltk
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# --------------------------------------------------
# NLTK DATA
# --------------------------------------------------

nltk.download("punkt_tab")
nltk.download("stopwords")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    "spam.csv",
    encoding="Windows-1252"
)


# Remove unnecessary columns
df.drop(
    columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    inplace=True
)


# Rename columns
df.rename(
    columns={
        "v1": "target",
        "v2": "text"
    },
    inplace=True
)


# --------------------------------------------------
# ENCODE TARGET
# ham  = 0
# spam = 1
# --------------------------------------------------

df["target"] = df["target"].map({
    "ham": 0,
    "spam": 1
})


# Remove duplicate messages
df = df.drop_duplicates(
    keep="first"
)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

ps = PorterStemmer()

STOP_WORDS = set(
    stopwords.words("english")
)


def transform_text(text):

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:

        if word.isalnum():

            cleaned_tokens.append(word)

    filtered_tokens = []

    for word in cleaned_tokens:

        if word not in STOP_WORDS and word not in string.punctuation:

            filtered_tokens.append(word)

    stemmed_tokens = []

    for word in filtered_tokens:

        stemmed_tokens.append(
            ps.stem(word)
        )

    return " ".join(stemmed_tokens)


# Apply preprocessing
df["transformed_text"] = df["text"].apply(
    transform_text
)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["transformed_text"]
)

y = df["target"]



model = MultinomialNB()

model.fit(
    X,
    y
)


joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


print("====================================")
print("Model training completed!")
print("====================================")
print("Model saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")
print("Total messages:", len(df))
print("====================================")