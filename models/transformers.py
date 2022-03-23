import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils import _IS_32BIT
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

from models.computer import WeightsComputer


class Transformer(TransformerMixin, BaseEstimator):

    def __init__(
        self, 
        weight_method: str='idf', 
        norm: str="l2",
        use_idf: bool=True,
        smooth_idf: bool=True,
        sublinear_tf: bool=False,
        tf_func: str=None
    ):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.weight_method = weight_method,
        self.tf_func = tf_func

    @staticmethod
    def _document_frequency(X):
        """Count the number of non-zero values for each feature in sparse X."""
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(X.indptr)

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        y : None
            This parameter is not needed to compute tf-idf.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # large sparse data is not supported for 32bit platforms because
        # _document_frequency uses np.bincount which works on arrays of
        # dtype NPY_INTP which is int32 for 32bit platforms. See #20923
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
        self.dtype = dtype

        if self.use_idf:
            n_features = X.shape[1]
            idf = self.get_collection_frequency_factor(X, y)

            if hasattr(self.computer, 'icf_mean'):
                self.icf_mean  = self.computer.icf_mean

            if hasattr(self.computer, 'icsdf_mean' ):
                self.icsdf_mean  = self.computer.icsdf_mean 

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation.
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        
        X.data = self.get_tf(X.data)

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")

            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.
        Returns
        -------
        ndarray of shape (n_features,)
        """
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(
            value, 
            diags=0, 
            m=n_features, 
            n=n_features, 
            format="csr"
        )

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}

    def get_collection_frequency_factor(self, X, y):
        """Computes Collection Frequency factor (IDF)
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.
        y : vector of class labels (n_samples,)
        dtype : data type of X matrix

        Availiable functions:
        ---------------------
        n_t - Number of times a term occured in the document
        n - number of words in document
        df(t) - document frequency of term t

        idf : log( (1 + n) / (1 + df(t)) ) + 1
        dfs: Distinguishing feature selector
        chi2 : Term Weighting Based on Chi-Square Statistic
        ig: Term weighting based on information gain
        igm : Term Weighting Based on Inverse Gravity Moment
        pb : Probability-Based Term Weighting
        idf_icf: Term Weighting Based on Inverse Class Frequency
        rf : Term Weighting Based on Relevance Frequency
        idf_icdf: Term Weighting Based on Inverse Class Density Frequency
        iadf : inverse average document frequency
        iadf_norm : inverse average document frequency normalized

        Returns
        -------
        Term-Weights matrix of shape X.shape
        
        """
        if type(self.weight_method) is tuple:
            weight_method = self.weight_method[0]
        else:
            weight_method = self.weight_method

        if type(self.dtype) is tuple:
            dtype = self.dtype[0]
        else:
            dtype = self.dtype

        self.computer = WeightsComputer(
            dtype = dtype,
            weight_method = weight_method,
            smooth_idf = self.smooth_idf
        )

        return self.computer.method(X, y)

    def get_tf(self, X):
        """Compute TF term:
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.

        Availiable functions:
        ---------------------
        n_t - Number of times a term occured in the document
        n - number of words in document

        'tf': n_t (standart realization of TF in sklearn), 
        'log_tf': log(n_t) + 1, 
        'sqrt_tf': sqrt(n_t)

        Returns
        -------
        TF matrix of shape X.shape
        """
        if self.tf_func=='tf':
            return X
        elif self.tf_func=='log_tf':
            return np.log(X + 1) 
        elif self.tf_func=='sqrt_tf':
            return np.sqrt(X)
        else:
            print('Wrong tf_func parameter values. Avaliable TF functions: ')
            print('tf, log_tf, sqrt_tf')
            raise ValueError