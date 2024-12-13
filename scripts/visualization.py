import matplotlib.pyplot as plt
import seaborn as sns

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
