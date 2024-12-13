import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")

def plot_distribution(data, column, title, xlabel, ylabel, bins=30, color='blue'):
    """Plot a histogram distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_bar(data, title, xlabel, ylabel, color='orange'):
    """Plot a bar chart."""
    data.plot(kind='bar', color=color, figsize=(12, 8))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def plot_sentiment_distribution(df, score_column):
    """
    Plots the distribution of sentiment scores.

    Parameters:
    - df (pd.DataFrame): The dataset containing the sentiment scores.
    - score_column (str): The column name containing sentiment scores.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[score_column], kde=True, bins=30, color='purple')
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

def plot_sentiment_categories(df, category_column):
    """
    Plots the sentiment categories.

    Parameters:
    - df (pd.DataFrame): The dataset containing the sentiment categories.
    - category_column (str): The column name containing sentiment categories.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=category_column, palette='viridis')
    plt.title("Sentiment Categories")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Articles")
    plt.show()

def plot_word_cloud(text):
    """
    Generates and displays a word cloud.

    Parameters:
    - text (str): The combined text for the word cloud.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Headlines")
    plt.show()
