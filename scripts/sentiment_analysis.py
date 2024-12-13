from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

def compute_vader_sentiment(df, text_column):
    """Compute sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df[text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

def categorize_sentiment(score):
    """Categorize sentiment score into positive, negative, or neutral."""
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def compute_textblob_sentiment(df, text_column):
    """Compute sentiment using TextBlob."""
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    return df
