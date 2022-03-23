import pandas as pd
import pymorphy2
import re

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class TextPreprocessor(BaseEstimator, TransformerMixin):  
    def __init__(self, stopwords:List=[]):
        self.morph = pymorphy2.MorphAnalyzer()
        self._stop_words = " ".join(stopwords + ['', ' '])
        self._cash = dict()
       
    def build_tokenizer(self, token_pattern: str = r"[а-яА-Яa-zA-Z\d-]{3,}"):
        return re.compile(token_pattern).findall
   
    def lemmatize(self, word):
        return self.morph.parse(word)[0].normal_form
   
    def text_preprocessor(self, text: str) -> str:
      
        tokens_lemmatized = []
        tokenizer = self.build_tokenizer()
        for token in tokenizer(text):
            if token in self._cash.keys():
                tokens_lemmatized.append(self._cash[token])
            else:
                lemma = self.lemmatize(token).strip()
                if lemma in self._stop_words:
                    continue
                tokens_lemmatized.append(lemma)
                self._cash[token] = lemma
           
            
        return " ".join(tokens_lemmatized)
    
    def fit(self, X , y=None):
        return self
   
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_list()
        for sentence in X:
            sentence = re.sub('[\n\t]+', ' ', sentence)
            sentence = re.sub('\s+', ' ', sentence)
            yield self.text_preprocessor(sentence)
    