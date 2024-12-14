import re

def extract_domain(email):
    match = re.search(r'@([\w.-]+)', email)
    return match.group(1) if match else 'Unknown'

def get_top_publishers(data, column, top_n=10):
    publisher_counts = data[column].value_counts()
    return publisher_counts.head(top_n)

def plot_top_publishers(publisher_counts):
    publisher_counts.plot(kind='bar', figsize=(12, 6), color='skyblue')
    plt.title('Top Publishers Contributing to the News Feed')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.show()
