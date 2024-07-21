import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, confusion_matrix
import re
import streamlit as st
import seaborn as sns

# Load dataset
aspects_df = pd.read_csv('semeval2014-1.csv')

# Aspect labels
aspect_labels = ["service", "food", "anecdotes/miscellaneous", "price", "ambience"]

# Function to handle contractions and negations
def preprocess_text(text):
    # Define contractions
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "shan't": "shall not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }

    # Replace contractions
    for contraction, replacement in contractions.items():
        text = re.sub(contraction, replacement, text)

    # Handle negations
    negations = ["not", "never", "no", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "cannot"]
    text_tokens = text.split()
    for i in range(len(text_tokens) - 1):
        if text_tokens[i] in negations:
            text_tokens[i + 1] = "NOT_" + text_tokens[i + 1]

    return " ".join(text_tokens)

# Apply preprocessing to text data
aspects_df["text"] = aspects_df["text"].apply(preprocess_text)

# Extract features and labels for aspect classification
X_aspect = aspects_df["text"]
y_aspect = np.asarray(aspects_df[aspects_df.columns[1:6]])

# Vectorize the text data for aspect classification
vectorizer_aspect = TfidfVectorizer(max_features=3000, max_df=0.85, ngram_range=(1,2))
X_aspect_tfidf = vectorizer_aspect.fit_transform(X_aspect)

# Split the dataset into training and testing sets for aspect classification
X_train_aspect, X_test_aspect, y_train_aspect, y_test_aspect = train_test_split(X_aspect_tfidf, y_aspect, test_size=0.33, random_state=42)

# Initialize SVM classifier for aspect classification
svm_classifier_aspect = SVC(kernel='linear')  # You can specify different kernels if needed

# Wrap SVM classifier in MultiOutputClassifier for multi-label classification
multi_label_classifier_aspect = MultiOutputClassifier(svm_classifier_aspect, n_jobs=-1)

# Train the classifier for aspect classification
multi_label_classifier_aspect.fit(X_train_aspect, y_train_aspect)

# Extract features and labels for sentiment analysis
X_sentiment = aspects_df["text"]
y_sentiment = aspects_df["review"]

# Vectorize the text data for sentiment analysis using bag-of-words model with n-grams
vectorizer_sentiment = CountVectorizer(max_features=3000, max_df=0.85, ngram_range=(1,2))
X_sentiment_bow = vectorizer_sentiment.fit_transform(X_sentiment)

# Split the dataset into training and testing sets for sentiment analysis
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X_sentiment_bow, y_sentiment, test_size=0.33, random_state=42)

# Initialize SVM classifier for sentiment analysis
svm_classifier_sentiment = SVC(kernel='linear')  # You can specify different kernels if needed

# Train the classifier for sentiment analysis
svm_classifier_sentiment.fit(X_train_sentiment, y_train_sentiment)

# Predict and evaluate on the test set for aspect classification
predicted_aspect = multi_label_classifier_aspect.predict(X_test_aspect)
print(f"Aspect Classification Accuracy: {accuracy_score(y_test_aspect, predicted_aspect)}")
print(f"Aspect Classification Hamming Loss: {hamming_loss(y_test_aspect, predicted_aspect)}")
print(f"Aspect Classification Precision: {precision_score(y_test_aspect, predicted_aspect, average='micro')}")

# Predict and evaluate on the test set for sentiment analysis
predicted_sentiment = svm_classifier_sentiment.predict(X_test_sentiment)
print(f"Sentiment Analysis Accuracy: {accuracy_score(y_test_sentiment, predicted_sentiment)}")
print(f"Sentiment Analysis Precision: {precision_score(y_test_sentiment, predicted_sentiment, average='micro', zero_division=1)}")

# Plot confusion matrix for sentiment analysis
conf_matrix_sentiment = confusion_matrix(y_test_sentiment, predicted_sentiment)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_sentiment, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Sentiment Analysis')
plt.show()

# Plot confusion matrix for aspect classification
for i, aspect in enumerate(aspect_labels):
    conf_matrix_aspect = confusion_matrix(y_test_aspect[:, i], predicted_aspect[:, i])
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_aspect, annot=True, fmt='d', cmap='Blues', xticklabels=['Not ' + aspect, aspect], yticklabels=['Not ' + aspect, aspect])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {aspect} Aspect')
    plt.show()

# Function to predict aspects and sentiment for new sentences
def predict_aspects_and_sentiment(new_sentences):
    # Apply preprocessing to new sentences
    new_sentences = [preprocess_text(sentence) for sentence in new_sentences]

    new_sentences_aspect_tfidf = vectorizer_aspect.transform(new_sentences)
    predicted_aspects = multi_label_classifier_aspect.predict(new_sentences_aspect_tfidf)
    
    new_sentences_sentiment_bow = vectorizer_sentiment.transform(new_sentences)
    predicted_sentiments = svm_classifier_sentiment.predict(new_sentences_sentiment_bow)
    
    # Convert numerical predictions to aspect labels
    predicted_aspect_labels = []
    for aspects in predicted_aspects:
        labels = [aspect_labels[i] for i, val in enumerate(aspects) if val == 1]
        predicted_aspect_labels.append("|".join(labels))
    
    return predicted_aspects, predicted_sentiments, predicted_aspect_labels

# Streamlit app
st.title('Hotel Review Aspect and Sentiment Analysis')

input_text = st.text_area('Enter a review:')
if st.button('Analyze'):
    if input_text:
        predicted_aspects, predicted_sentiments, predicted_aspect_labels = predict_aspects_and_sentiment([input_text])
        st.write(f"Predicted Aspects: {predicted_aspects}")
        st.write(f"Predicted Sentiments: {predicted_sentiments}")
        st.write(f"Predicted Aspect Labels: {predicted_aspect_labels}")
    else:
        st.write("Please enter a review.")