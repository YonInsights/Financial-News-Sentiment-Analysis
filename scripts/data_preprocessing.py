import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)
# Function to preprocess sentiment data
def preprocess_sentiment_data(file_path, text_column):
    news_df = pd.read_csv(file_path)

    # Analyze sentiment
    news_df = analyze_sentiment_vader(news_df, text_column)

    # Aggregate daily sentiment scores
    daily_sentiment = aggregate_daily_sentiment(news_df, 'date', 'sentiment_score')

    return daily_sentiment

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(df, text_column):
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

# Modified function to aggregate daily sentiment
def aggregate_daily_sentiment(df, date_column, sentiment_column):
    df[date_column] = pd.to_datetime(df[date_column], format="%Y-%m-%d %H:%M:%S")
    daily_sentiment = df.groupby(df[date_column].dt.date)[sentiment_column].mean().reset_index()
    daily_sentiment.columns = ['date', 'average_sentiment']
    return daily_sentiment
