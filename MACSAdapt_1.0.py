
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd


print("\nImporting csv file columns into Python with Pandas")
df = pd.read_csv('macsadapt.csv', encoding="ISO-8859-1")
print(df)

print("\nVectorizing the list of Abstracts into one sparse matrix")
saved_columns = df.ABSTRACT
abstracts = df.ABSTRACT
ListAbstracts = df.ABSTRACT.values.tolist()

#removing the carriage return "\r" from string
# TODO Improve the cleaning of documents
# Remove undesired elements in List Abstracts
ListAbstracts2 = [item.replace("\r", "") for item in ListAbstracts]
ListAbstracts3 = [item.replace(",", "") for item in ListAbstracts2]
ListAbstracts4 = [item.replace(".", "") for item in ListAbstracts3]

print(ListAbstracts4)


#Stemming the words in the Abstracts
from stemming.porter2 import stem
ListAbstracts5 = [[stem(word) for word in sentence.split(" ")] for sentence in ListAbstracts4]
ListAbstracts6 = [" ".join(sentence) for sentence in ListAbstracts5]
print(ListAbstracts6)

#Modelizing topics from the database

n_samples = 2276
n_features = 1000
n_components = 20
n_top_words = 6

data = ListAbstracts6
data_samples = data[:n_samples]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


print("\nExtracting tf-idf features for NMF")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print(tfidf)

print("\nExtracting tf features for Latent Dirichlet Allocation")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)

print("\nFitting the NMF model (Frobenius norm) with tf-idf features")
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("\nFitting the NMF model (generalized Kullback-Leibler divergence) with tf-idf features")
nmf = NMF(n_components=n_components, random_state=1,beta_loss='kullback-leibler',
          solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5).fit(tfidf)

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names= tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("\nFitting LDA models with tf features, n_samples, n_features")
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

doc_topic_distrib = lda.transform(tf)
print(doc_topic_distrib)


# Log Likelihood: Higher the better
print("Log Likelihood: ", lda.score(tfidf))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda.perplexity(tfidf))
