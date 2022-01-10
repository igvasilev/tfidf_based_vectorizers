import numpy as np 
import scipy.sparse as sp
from sklearn.utils.fixes import _astype_copy_false


class WeightsComputer:
    '''
    Weight methods:
        idf : log( (1 + n) / (1 + df(t)) ) + 1
        dfs : Distinguishing feature selector
        chi2 : Term Weighting Based on Chi-Square Statistic
        ig : Term weighting based on information gain
        igm: TODO
        pb : Probability-Based Term Weighting
        idf_icf : Term Weighting Based on Inverse Class Frequency
        rf : Term Weighting Based on Relevance Frequency
        idf_icsdf : TODO
        iadf : inverse average document frequency
        iadf_norm : inverse average document frequency normalized
    '''
    def __init__(
        self, 
        dtype, 
        weight_method:str, 
        smooth_idf: bool = True
    ):
        try:
            if type(weight_method) is tuple:
                weight_method = weight_method[0]
            self.method = getattr(self, weight_method)
        except AttributeError:
            print(f'Method {weight_method} is not implemmnted.')
            print('Check the list of avaliable parameters')
        
        self.dtype = dtype 
        if type(self.dtype) is tuple:
            self.dtype = self.dtype[0]

        self.smooth_idf = smooth_idf
        self.cross_tab = None

    @staticmethod
    def _document_frequency(X):
        """Count the number of non-zero values for each feature in sparse X."""
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(X.indptr)

    def make_cross_tab(self, X, y):
        '''Computes Two-way contingency table of a term t
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.
        y : vector of class labels (n_samples,)

        Returns
        -------
        np.array of shape (n_classes, 4, n_features)
        '''
        cross_tab = []
        classes = np.unique(y)
        for cls in classes:
            cat = np.where(y==cls)[0]
            not_cat = np.where(y!=cls)[0]
            # Belong to cls, contain term t
            a = self._document_frequency(X[cat])
            # Belong to cls, doesn`t contain term t
            b = np.sum(cat) - a
            # Don`t belong to cls, contain term t
            c = self._document_frequency(X[not_cat])
            # Don`t belong to cls, doesn`t contain term t
            d = np.sum(not_cat) - c

            cross_tab.append([a,b,c,d])

        self.cross_tab = cross_tab


    def idf(self, X, y):
        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        n_docs= X.shape[0]
        df = self._document_frequency(X)
        df = df.astype(self.dtype, **_astype_copy_false(df))
        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_docs += int(self.smooth_idf)
        return np.log(n_docs / df) + 1

    def dfs(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
            nominator = a/(a + c)
            denominator = (b/(a+b)) + (c/(c+d)) + 1
            weight_factors.append(nominator / denominator)
        return np.sum(weight_factors, axis=0) 

    def chi2(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        # N documents
        D = X.shape[0]
        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
            nominator = np.square(a*d -b*c)
            denominator = (a+c)*(b+d)*(a+b)*(c+d)
            weight_factors.append(nominator / denominator)
        return D * np.max(weight_factors, axis=0) 

    def ig(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        # N documents
        N = X.shape[0]
        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
            # Add +1 to denominators to avoid Zero Division error
            first = a/N*np.log(1+(a*N)/((a+c)*(a+b)+1))
            second = b/N*np.log(1+(b*N)/((b+d)*(a+b)+1))
            third = c/N*np.log(1+(c*N)/((a+c)*(c+d)+1))
            fourth = d/N*np.log(1+(d*N)/((b+d)*(c+d)+1))
            weight_factors.append(first + second + third + fourth)
        return np.max(weight_factors, axis=0) 

    def igm(self, X, y):
        raise NotImplementedError

    def pb(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
            # Add +1 to denominator to avoid Zero Division error
            first =  a / (np.max(np.c_[b, np.ones_like(b)], axis=1))
            second = a / (np.max(np.c_[c, np.ones_like(c)], axis=1))
            pb = np.log(1 + first*second)
            weight_factors.append(pb)
        return np.max(weight_factors, axis=0) 

    def idf_icf(self, X, y):
        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        n_docs = X.shape[0]
        df = self._document_frequency(X)
        df = df.astype(self.dtype, **_astype_copy_false(df))
        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_docs += int(self.smooth_idf)
        idf =  np.log(n_docs / df) + 1

        n_classes = len(np.unique(y))
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        # Number of classes where term t occures
        class_factors = np.zeros_like(self.cross_tab[0][0])
        for category in self.cross_tab:
            a, b, c, d = category
            class_factors += (a > 0)
        
        icf = np.log(n_classes/class_factors) + 1
        return idf*icf

    def rf(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
            rf = np.log(2 + a / (np.max(np.c_[c, np.ones_like(c)], axis=1)))
            weight_factors.append(rf)
        return np.max(weight_factors, axis=0) 

    def idf_icsdf(self, X, y):
        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        n_docs = X.shape[0]
        df = self._document_frequency(X)
        df = df.astype(self.dtype, **_astype_copy_false(df))
        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_docs += int(self.smooth_idf)
        idf =  np.log(n_docs / df) + 1

        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        class_factors = []
        for i, category in enumerate(self.cross_tab):
            a, b, c, d = category
            D_cls = counts[i]
            class_factors.append(a / D_cls)

        icsdf = np.log(n_classes / np.sum(class_factors, axis=0)) + 1
        return idf * icsdf

    def iadf(self, X, y):
        D = X.shape[0]
        n_terms = X.shape[1] 
        df = self._document_frequency(X)
        df = df.astype(self.dtype, **_astype_copy_false(df))
        mean_df = np.sum(df) / n_terms
        adf = np.square(df - mean_df) / n_terms
        return np.log((D + 1) / (adf + 1))

    def iadf_norm(self, X, y):
        D = X.shape[0]
        n_terms = X.shape[1] 
        df = self._document_frequency(X)
        df = df.astype(self.dtype, **_astype_copy_false(df))
        mean_df = np.sum(df) / n_terms
        adf = np.square(df - mean_df) / n_terms
        adf_1 = np.log(1/(adf+1)) + 1
        adf_2 = (adf_1 - np.min(adf_1)) / (np.max(adf_1) - np.min(adf_1))
        return np.log((D+1) / (adf_2 + 1))