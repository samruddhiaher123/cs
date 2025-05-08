import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Download stopwords
nltk.download('stopwords')

# Load data
df = pd.read_csv('C:/users/asus/spam.csv', encoding='latin-1')
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df[['label', 'text']]

# Add a column for message length
df['length'] = df['text'].apply(len)

# Plot message length by label
df.hist(column='length', by='label', bins=50, figsize=(10, 4))
plt.tight_layout()
plt.show()

# Prepare stopwords once for efficiency
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def clean_text(text):
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    return [w.lower() for w in words if w.lower() not in stop_words]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Create pipeline with vectorizer, tf-idf, and Naive Bayes classifier
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer=clean_text)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)
import seaborn as sns

# Histogram with KDE (Kernel Density Estimation)
sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
plt.title('Distribution of Message Length by Label')
plt.show()

# Alternatively, use boxplot for spread and outlier detection
sns.boxplot(x='label', y='length', data=df)
plt.title('Boxplot of Message Length by Label')
plt.show()

# Show classification report
print(classification_report(y_test, pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.show()
