Step 1: Data Retrieval

1.1	Access the provided Google Sheets link to obtain the dataset.

Step 2: Data Preprocessing

2.1	Load the dataset into a pandas DataFrame for easier manipulation.

```python
import pandas as pd

# Load the dataset url =
"https://docs.google.com/spreadsheets/d/1DBSBx8S9I1Ug2DMmWrDQHta648htciBua_lVs psC8pU/export?format=csv"
df = pd.read_csv(url)

# Verify the loaded data print(df.head())
```

2.2	Perform text cleaning using Nltk or any other library.

```python import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize from nltk.stem import PorterStemmer from bs4 import BeautifulSoup
import re

# Download NLTK resources nltk.download('punkt') nltk.download('stopwords')

# Text cleaning function def clean_text(text):
soup = BeautifulSoup(text, 'html.parser') cleaned_text = soup.get_text()
cleaned_text = re.sub(r"[^a-zA-Z]", " ", cleaned_text) # Remove non-alphabetic characters words = word_tokenize(cleaned_text.lower()) # Tokenization and lowercase conversion words = [word for word in words if word not in stopwords.words('english')] # Remove
stopwords
stemmer = PorterStemmer()
words = [stemmer.stem(word) for word in words] # Perform stemming cleaned_text = ' '.join(words)
return cleaned_text
 
# Apply text cleaning to the 'text' column of the DataFrame df['cleaned_text'] = df['text'].apply(clean_text)
```

Step 3: Feature Extraction

3.1	Choose a feature extraction technique (e.g., TF-IDF) and apply it to the cleaned text data.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust the number of features as needed
tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
```

Step 4: Model Building

4.1	Choose suitable machine learning algorithms (e.g., Naive Bayes, SVM, Random Forest) and build models.

```python
from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['label'], test_size=0.2, random_state=42)

# Build and train Naive Bayes classifier nb_classifier = MultinomialNB() nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted') recall = recall_score(y_test, predictions, average='weighted')

print("Accuracy:", accuracy) print("Precision:", precision) print("Recall:", recall)
```

Step 5: Visualization
 
5.1	Create a word cloud representing the top 20 most frequent words.

```python
from wordcloud import WordCloud import matplotlib.pyplot as plt

# Get the most frequent words
word_freq = tfidf_vectorizer.vocabulary_
top_20_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_20_words)

# Display the word cloud using matplotlib plt.figure(figsize=(10, 5)) plt.imshow(wordcloud, interpolation='bilinear') plt.axis("off")
plt.show()
```

