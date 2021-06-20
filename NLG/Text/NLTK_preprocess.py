# pre-processing script
# cleans text & generates corpus statistics
# useful for further NLP
import re
import csv

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# data_file should contain your data as a pandas data frame
data_file = '/Users/.../result_pd.csv'

# print indicates one of the data frame columns
file_prefix = data_file.split('.')
file_prefix = file_prefix[0] + '_'
print('\nFile exports will be prefixed with:', file_prefix)
dataset = pd.read_csv(data_file, delimiter=None)
print(dataset['subt'])


# data column indicated with dataset['subt']
# print[datacol] for test
freq = pd.Series(' '.join(map(str, dataset['subt'])).split()).value_counts()[:10]
# freq

freq1 = pd.Series(' '.join(map(str, dataset['subt'])).split()).value_counts()[-10:]
# freq1

stop_words = set(stopwords.words("english"))
print(sorted(stop_words))

# load stopword file generated after prelim results
csw = set(line.strip() for line in open('custom-stopwords.txt'))
csw = [sw.lower() for sw in csw]
print(sorted(csw))

# Combine custom stop words with stop_words list
stop_words = stop_words.union(csw)
print(sorted(stop_words))
# build corpus
corpus = []
dataset['word_count'] = dataset['subt'].apply(lambda x: len(str(x).split(" ")))
ds_count = len(dataset.word_count)
dataset[['video_title','subt','word_count']].head()
print(dataset[['subt','word_count']])

# grab descriptive and word count
print(dataset.word_count.describe())
count_df = pd.DataFrame()
count_df['text'] = dataset['subt']
count_df['word_count'] = dataset['word_count']
print(sum(count_df['word_count']))
count_df.to_csv('results_pd_word_count.csv')


for i in range(0, ds_count):
    # Remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', str(dataset['subt'][i]))

    # Convert to lowercase
    text = text.lower()

    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    # Convert to list from string
    text = text.split()

    # Stemming
    ps = PorterStemmer()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    corpus.append(text)

print(corpus[ds_count - 131])
#
# matplotlib inline
wordcloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    max_words=100,
    max_font_size=50,
    random_state=42
).generate(str(corpus))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig(file_prefix + "wordcloud.png", dpi=3600)

# Tokenize the text and build a vocabulary of known words
cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))
X = cv.fit_transform(corpus)

# Sample the returned vector encoding the length of the entire vocabulary
list(cv.vocabulary_.keys())[:10]

# View most frequently occuring keywords
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# Convert most freq words to dataframe for plotting bar plot, save as CSV
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns = ["Keyword", "Frequency"]
print(top_df)
top_df.to_csv(file_prefix + '_top_words.csv')

# Barplot of most freq words
sns.set(rc={'figure.figsize': (13, 8)})
g = sns.barplot(x="Keyword", y="Frequency", data=top_df, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.figure.savefig(file_prefix + "_keyword.png", bbox_inches="tight")


# Most frequently occuring bigrams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2),
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]

# Convert most freq bigrams to dataframe for plotting bar plot, save as CSV
top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns = ["Bi-gram", "Frequency"]
print(top2_df)
top2_df.to_csv(file_prefix + '_bigrams.csv')

# Barplot of most freq Bi-grams
sns.set(rc={'figure.figsize': (13, 8)})
h = sns.barplot(x="Bi-gram", y="Frequency", data=top2_df, palette="Blues_d")
h.set_xticklabels(h.get_xticklabels(), rotation=75)
h.figure.savefig(file_prefix + "_bi-gram.png", bbox_inches="tight")


# Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3, 3),
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# Convert most freq trigrams to dataframe for plotting bar plot, save as CSV
top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns = ["Tri-gram", "Frequency"]
print(top3_df)
top3_df.to_csv(file_prefix + '_trigrams.csv')

# Barplot of most freq Tri-grams
sns.set(rc={'figure.figsize': (13, 8)})
j = sns.barplot(x="Tri-gram", y="Frequency", data=top3_df, palette="Blues_d")
j.set_xticklabels(j.get_xticklabels(), rotation=75)
j.figure.savefig(file_prefix + "_tri-gram.png", bbox_inches="tight")

# Get TF-IDF (term frequency/inverse document frequency) --
# TF-IDF lists word frequency scores that highlight words that
# are more important to the context rather than those that
# appear frequently across documents
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)

# Get feature names
feature_names = cv.get_feature_names()

# Fetch document for which keywords needs to be extracted
doc = corpus[ds_count - 131]


# Generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))


# Sort tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=25):
    # Use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # Word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # Keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # Create tuples of feature,score
    # Results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


# Sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())

# Extract only the top n; n here is 25
keywords = extract_topn_from_vector(feature_names, sorted_items, 25)

# Print the results, save as CSV
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])

# write csv
# import csv
with open(file_prefix + 'td_idf.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Keyword", "Importance"])
    for key, value in keywords.items():
        writer.writerow([key, value])
