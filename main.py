import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_loader import load_data
from scripts.data_preprocessing import preprocess_text
from scripts.descriptive_statistics import compute_headline_length, compute_basic_statistics, find_outliers
from scripts.sentiment_analysis import compute_vader_sentiment, compute_textblob_sentiment
from scripts.visualization import plot_distribution, plot_bar

# Load data
file_path = r"D:\Kifya_training\Week 1\Technical  Content\Data\raw_analyst_ratings.csv"
df = load_data(file_path)

# Preprocess data
df['cleaned_text'] = df['headline'].apply(preprocess_text)

# Perform descriptive analysis
df = compute_headline_length(df)
basic_stats = compute_basic_statistics(df)
outliers = find_outliers(df, 'headline_length')

# Perform sentiment analysis
df = compute_vader_sentiment(df, 'headline')
df = compute_textblob_sentiment(df, 'cleaned_text')

# Visualize results
plot_distribution(df, 'headline_length', "Distribution of Headline Lengths", "Headline Length", "Frequency")
plot_bar(df['publisher'].value_counts().head(10), "Top 10 Most Active Publishers", "Publisher", "Number of Articles")
