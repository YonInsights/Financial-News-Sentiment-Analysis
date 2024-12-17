from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
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

# New functions for Task 3
def analyze_sentiment_vader(df, text_column):
    """
    Analyze sentiment of text data using VADER.
    Args:
        df (pd.DataFrame): Dataframe containing the text column.
        text_column (str): Name of the text column to analyze.
    Returns:
        pd.DataFrame: Dataframe with sentiment scores added.
    """
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def aggregate_daily_sentiment(df, date_column, sentiment_column):
    """
    Aggregates average daily sentiment scores.

    Args:
    - df (pd.DataFrame): The dataframe containing sentiment scores.
    - date_column (str): The name of the date column.
    - sentiment_column (str): The name of the sentiment score column.

    Returns:
    - pd.DataFrame: A dataframe with daily average sentiment scores.
    """
    # Ensure the date column is a datetime type
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Remove any rows with invalid dates
    df = df.dropna(subset=[date_column])

    # Group by date and compute average sentiment
    daily_sentiment = df.groupby(df[date_column].dt.date)[sentiment_column].mean().reset_index()
    daily_sentiment.columns = ['date', 'average_sentiment']

    return daily_sentiment




def preprocess_sentiment_data(file_path, text_column):
    """
    Loads, analyzes, and aggregates sentiment data from a news dataset.

    Parameters:
    - file_path (str): Path to the news CSV file.
    - text_column (str): The column containing news headlines.

    Returns:
    - pd.DataFrame: Preprocessed and aggregated daily sentiment data.
    """
    # Load data
    news_df = pd.read_csv(file_path)
    
    # Analyze sentiment
    news_df = analyze_sentiment_vader(news_df, text_column)
    
    # Aggregate daily sentiment scores
    daily_sentiment = aggregate_daily_sentiment(news_df, 'date', 'sentiment_score')
    
    return daily_sentiment

# Set plot style
sns.set(style="whitegrid")

def plot_daily_sentiment(daily_sentiment):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=daily_sentiment, x='date', y='average_sentiment', marker='o', color='blue')
    plt.title('Daily Average Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Plot the daily average sentiment
plot_daily_sentiment(daily_sentiment)

