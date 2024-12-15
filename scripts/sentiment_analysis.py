from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

def initialize_vader():
    """
    Initializes the VADER sentiment intensity analyzer.

    Returns:
    - SentimentIntensityAnalyzer: The VADER sentiment analyzer object.
    """
    from nltk import download
    download('vader_lexicon')
    return SentimentIntensityAnalyzer()

def analyze_sentiment_vader(df, text_column):
    """
    Analyzes sentiment using VADER and categorizes the results.

    Parameters:
    - df (pd.DataFrame): The dataset containing the text column.
    - text_column (str): The column name containing text to analyze.

    Returns:
    - pd.DataFrame: The dataset with sentiment scores and categories.
    """
    # Initialize VADER
    sia = initialize_vader()
    
    # Compute sentiment polarity using VADER
    df['sentiment_score'] = df[text_column].fillna("").apply(lambda x: sia.polarity_scores(x)['compound'])

    # Categorize sentiment
    def categorize_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    
    return df

def analyze_sentiment_textblob(df, text_column):
    """
    Analyzes sentiment using TextBlob.

    Parameters:
    - df (pd.DataFrame): The dataset containing the text column.
    - text_column (str): The column name containing text to analyze.

    Returns:
    - pd.DataFrame: The dataset with TextBlob sentiment scores and labels.
    """
    # Compute sentiment polarity using TextBlob
    df['sentiment'] = df[text_column].fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)

    # Classify sentiment
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    
    return df

def calculate_textblob_sentiment(data, column):
    data['textblob_sentiment'] = data[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data

def calculate_vader_sentiment(data, column):
    sia = SentimentIntensityAnalyzer()
    data['vader_sentiment'] = data[column].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data
