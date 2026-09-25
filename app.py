import streamlit as st
import joblib
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# --------------------------------------------------
# NLTK
# --------------------------------------------------

nltk.download(
    "punkt_tab",
    quiet=True
)

nltk.download(
    "stopwords",
    quiet=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    "model.pkl"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
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

    tokens = nltk.word_tokenize(
        text
    )

    cleaned_tokens = []

    for word in tokens:

        if word.isalnum():

            cleaned_tokens.append(
                word
            )

    filtered_tokens = []

    for word in cleaned_tokens:

        if (
            word not in STOP_WORDS
            and word not in string.punctuation
        ):

            filtered_tokens.append(
                word
            )

    stemmed_tokens = []

    for word in filtered_tokens:

        stemmed_tokens.append(
            ps.stem(word)
        )

    return " ".join(
        stemmed_tokens
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📧 Email & SMS Spam Classifier")

st.write(
    "Enter an email or SMS message "
    "to determine whether it is Spam or Ham."
)


st.divider()


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

message = st.text_area(
    "Enter your message",
    height=180,
    placeholder=(
        "Example: "
        "Congratulations! You have won a "
        "free prize. Click here to claim..."
    )
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Check Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

    else:

        # Preprocess
        transformed_message = transform_text(
            message
        )

        # TF-IDF transformation
        vectorized_message = vectorizer.transform(
            [transformed_message]
        )

        # Prediction
        prediction = model.predict(
            vectorized_message
        )[0]

        # Probability
        probabilities = model.predict_proba(
            vectorized_message
        )[0]

        ham_probability = probabilities[0] * 100
        spam_probability = probabilities[1] * 100


        st.divider()


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 SPAM MESSAGE"
            )

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

        else:

            st.success(
                "✅ NOT SPAM "
            )

            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )


        # --------------------------------------------------
        # PROBABILITY DETAILS
        # --------------------------------------------------

        st.subheader(
            "Prediction Details"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Ham:** {ham_probability:.2f}%"
            )

        with col2:

            st.write(
                f"**Spam:** {spam_probability:.2f}%"
            )


        st.progress(
            int(spam_probability)
        )