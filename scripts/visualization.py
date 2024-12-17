import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")

# Function to plot a histogram distribution
def plot_distribution(data, column, title, xlabel, ylabel, bins=30, color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Function to plot a bar chart
def plot_bar(data, title, xlabel, ylabel, color='orange'):
    data.plot(kind='bar', color=color, figsize=(12, 8))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

# Function to plot the distribution of sentiment scores
def plot_sentiment_distribution(df, score_column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[score_column], kde=True, bins=30, color='purple')
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

# Function to plot sentiment categories
def plot_sentiment_categories(df, category_column):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=category_column, palette='viridis')
    plt.title("Sentiment Categories")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Articles")
    plt.show()

# Function to generate and display a word cloud
def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Headlines")
    plt.show()

# Function to plot daily publication frequency with market events
def plot_daily_publication(daily_publication, market_events=None):
    plt.figure(figsize=(12, 6))
    daily_publication.plot()

    if market_events:
        for event in market_events:
            plt.axvline(x=event, color='r', linestyle='--', label=f"Market Event: {event.date()}")
    plt.title('Daily Publication Frequency with Market Events')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot hourly publication frequency
def plot_hourly_publication(hourly_publication):
    plt.figure(figsize=(12, 6))
    hourly_publication.plot(kind='bar', color='skyblue')
    plt.title('Hourly Publication Frequency')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Articles')
    plt.xticks(range(0, 24))
    plt.grid(axis='y')
    plt.show()

# Function to plot a heatmap of data points by day of week and hour
def plot_heatmap(data):
    heatmap_data = data.pivot_table(index='hour', columns='day_of_week', aggfunc='size', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='d')
    plt.title('Frequency of Data Points by Day of Week and Hour')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    plt.show()

# Function to plot stock with sma
def plot_stock_with_sma(df):
    """Plot stock prices with SMA."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA'], label='SMA (20)')
    plt.title("Stock Price with SMA")
    plt.legend()
    plt.show()


# Set plot style
def plot_daily_sentiment(sentiment_df):
    """
    Plot daily sentiment scores over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_df['date'], sentiment_df['average_sentiment'], marker='o')
    plt.title("Daily Sentiment Scores")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
