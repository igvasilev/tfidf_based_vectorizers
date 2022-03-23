import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

from models.transformers import Transformer


class Vectorizer(CountVectorizer):
    '''
    This class instantiates tf-idf based vectorizers

    List of availiable methods:

        Availiable TF functions:
        ---------------------
            n_t - Number of times a term occured in the document
            n - number of words in document

            'tf' : n_t (standart realization of TF in sklearn), 
            'log_tf': log(n_t) + 1, 
            'sqrt_tf': sqrt(n_t)

        Weight methods:
        ---------------------
            idf : log( (1 + n) / (1 + df(t)) ) + 1
            dfs : Distinguishing feature selector
            chi2 : Term Weighting Based on Chi-Square Statistic
            ig : Term weighting based on information gain
            igm: Term Weighting Based on Inverse Gravity Moment
            pb : Probability-Based Term Weighting
            idf_icf : Term Weighting Based on Inverse Class Frequency
            rf : Term Weighting Based on Relevance Frequency
            idf_icsdf : Term Weighting Based on Inverse Class Density Frequency
            iadf : inverse average document frequency
            iadf_norm : inverse average document frequency normalized
    '''

    def __init__(
        self,
        weight_method: str='idf',
        tf_func: str='tf',
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self.transformer = Transformer(
            weight_method=weight_method, 
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            tf_func=tf_func
        )


    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : None
            This parameter is not needed to compute tfidf.
        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self._check_params()
        self._warn_for_unused_params()
        X = super().fit_transform(raw_documents)
        self.transformer.fit(X, y)

        if hasattr(self.transformer, 'icf_mean'):
            self.icf_mean  = self.transformer.icf_mean

        if hasattr(self.transformer, 'icsdf_mean' ):
            self.icsdf_mean  = self.transformer.icsdf_mean 

        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : None
            This parameter is ignored.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """

        self._check_params()
        X = super().fit_transform(raw_documents)
        self.transformer.fit(X, y)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        
        if hasattr(self.transformer, 'icf_mean'):
            self.icf_mean  = self.transformer.icf_mean

        if hasattr(self.transformer, 'icsdf_mean' ):
            self.icsdf_mean  = self.transformer.icsdf_mean 

        return self.transformer.transform(X, copy=False)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """

        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")
        X = super().transform(raw_documents)
        return self.transformer.transform(X, copy=False)

    # TODO: Add broadcasting for Multipliers

    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}