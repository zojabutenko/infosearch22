from transformers import AutoModel, AutoTokenizer
from nltk.corpus import stopwords
from scipy import sparse
import numpy as np
import pymorphy2
import re
import pickle
import torch


stopwords = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

answers = np.load('models/answers.npy')

clean_questions = np.load('models/questions.npy')

# import matrices:
# BM25
questions_bm25 = sparse.load_npz('models/bm25/questions.npz')

# BERT
model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
questions_bert = sparse.load_npz('models/bert/questions.npz')


# TF-IDF
with open('models/vectorizers/TfIdf_Vectorizer.pk', 'rb') as fp:
    vectorizer = pickle.load(fp)
questions_tfidf = sparse.load_npz('models/tf_idf/questions.npz')

# preprocess (lemmatize, get lowercase, remove stopwords and punctuation)
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = []
    for token in text.split():
        token = token.lower()
        if token.isalpha() and token not in stopwords:
            token = morph.normal_forms(token.strip())[0]
            tokens.append(token)
    return ' '.join(tokens)


# indexing query, returns query vector
def query_index(query, mode):
    if mode == 'bert':
        t = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return sparse.csr_matrix(embeddings[0].cpu().numpy())
    elif mode == 'bm25' or mode == 'tfidf':
        preprocessed_query = preprocess(query)
        query_vector = vectorizer.transform([preprocessed_query])
        return query_vector


# get documents relevant to the query
def search(query, mode):
    query = query_index(query, mode)
    matrices = {'bm25':  questions_bm25,
                'tfidf': questions_tfidf,
                'bert':  questions_bert}
    result_matrix = np.dot(matrices[mode], query.T).toarray()
    scores = np.argsort(result_matrix, axis=0)[::-1][:5]
    return [(answers[i], clean_questions[i]) for i in scores.ravel()]
