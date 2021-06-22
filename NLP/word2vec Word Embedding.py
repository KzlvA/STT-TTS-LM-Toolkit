#!/usr/bin/env python
# coding: utf-8

# # `word2vec` Word Embedding
# 

# In[ ]:


# !python -m spacy download en_core_web_lg


# ## Word Vectors with Spacy

# https://github.com/explosion/spaCy

# https://spacy.io/usage/vectors-similarity

# Similarity is determined by comparing word vectors or “word embeddings”, multi-dimensional meaning representations of a word. 
# 
# `python -m spacy download en_core_web_lg`

# `en_vectors_web_lg`, which includes over 1 million unique vectors

# In[ ]:


import spacy


# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


x = 'dog cat lion dsfaf'
doc = nlp(x)


# In[ ]:


for token in doc:
    print(token.text, token.has_vector, token.vector_norm)


# ## Semantic Similarity 

# spaCy is able to compare two objects, and make a prediction of how similar they are. Predicting similarity is useful for building recommendation systems or flagging duplicates. 
# 
# For example, you can suggest a user content that’s similar to what they’re currently looking at, or label a support ticket as a duplicate if it’s very similar to an already existing one.

# Each `Doc, Span and Token` comes with a `.similarity()` method that lets you compare it with another object, and determine the similarity.

# In[ ]:


x


# In[ ]:


doc = nlp(x)


# In[ ]:


for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))


# In[ ]:





# # Model Building for `word2vec` 

# ## Data Preparation 

# In[ ]:


# !pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report


# In[ ]:


import preprocess_kgptalkie as ps


# In[ ]:


df = pd.read_csv('imdb_reviews.txt', sep = '\t', header = None)
df.columns = ['reviews', 'sentiment']


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


x = "A very, very, very slow-moving, aimlss movie"
ps.spelling_correction(x).raw_sentences[0]


# In[ ]:


get_ipython().run_cell_magic('time', '', "df['reviews'] = df['reviews'].apply(lambda x: ps.cont_exp(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.remove_emails(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.remove_html_tags(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.remove_urls(x))\n\ndf['reviews'] = df['reviews'].apply(lambda x: ps.remove_special_chars(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.remove_accented_chars(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.make_base(x))\ndf['reviews'] = df['reviews'].apply(lambda x: ps.spelling_correction(x).raw_sentences[0])")


# In[ ]:


df.head()


# ## ML Model Building 

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_lg')


# In[ ]:


x = 'cat dog'
doc = nlp(x)


# In[ ]:


doc.vector.shape


# In[ ]:


doc.vector.reshape(1, -1).shape


# In[ ]:


def get_vec(x):
    doc = nlp(x)
    vec = doc.vector
    return vec


# In[ ]:


df['vec'] = df['reviews'].apply(lambda x: get_vec(x))


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


X = df['vec'].to_numpy()
X = X.reshape(-1, 1)


# In[ ]:


X.shape


# In[ ]:


X = np.concatenate(np.concatenate(X, axis = 0), axis = 0).reshape(-1, 300)


# In[ ]:


X.shape


# In[ ]:


y = df['sentiment']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:





# ## ML Model Traning and Testing 

# In[ ]:


clf = LogisticRegression(solver = 'liblinear', )


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:


import pickle 


# In[ ]:


pickle.dump(clf, open('w2v_sentiment.pkl', 'wb'))


# In[ ]:





# ## Support Vector Machine on `word2vec`

# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


clf = LinearSVC()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:





# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:





# ## Grid Search Cross Validation for Hyperparameters Tuning¶ 

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


logit = LogisticRegression(solver = 'liblinear')


# In[ ]:


hyperparameters = {
    'penalty': ['l1', 'l2'],
    'C': (1, 2, 3, 4)
}


# In[ ]:


clf = GridSearchCV(logit, hyperparameters, n_jobs=-1, cv = 5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X_train, y_train)')


# In[ ]:


clf.best_params_


# In[ ]:


clf.best_score_


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:





# ## Test Every Machine Learning Model 

# https://pypi.org/project/lazypredict/

# In[ ]:


get_ipython().system('pip install lazypredict')


# In[ ]:


# !pip install xgboost
# !pip install lightgbm
# install it with terminal in admin mode


# In[ ]:


from lazypredict.Supervised import LazyClassifier


# In[ ]:


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'models, predictions = clf.fit(X_train, X_test,  y_train, y_test)')


# In[ ]:


models


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




