from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def perform_topic_modeling(data, column, n_topics=5, n_words=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(data[column])

    nmf = NMF(n_components=n_topics, random_state=1)
    nmf.fit(X)

    topics = []
    for index, topic in enumerate(nmf.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]]
        topics.append(top_words)

    return topics
