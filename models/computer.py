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
        self.igm_lambda = 7.0
        self.cross_tab = None

    @staticmethod
    def _document_frequency(X):
        """Count the number of non-zero values for each feature in sparse X."""
        # print('type X: ', type(X))
        print(X.shape)
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
        n_docs = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)
        for i, cls in enumerate(classes):
            cat = np.array(np.where(y==cls)).flatten()
            not_cat = np.array(np.where(y!=cls)).flatten()
            # Belong to cls, contain term t
            a = self._document_frequency(X[cat]) + int(self.smooth_idf)
            # Belong to cls, doesn`t contain term t
            b = counts[i] - a + 2*int(self.smooth_idf)
            # Don`t belong to cls, contain term t
            c = self._document_frequency(X[not_cat]) + int(self.smooth_idf)
            # Don`t belong to cls, doesn`t contain term t
            d = (n_docs - counts[i]) - c + 2*int(self.smooth_idf)

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
            nominator = a / (np.max(np.c_[a + c, np.ones_like(a + c)], axis=1))
            denom_first = np.max(np.c_[a + b, np.ones_like(a + b)], axis=1) 
            denom_second = np.max(np.c_[c+d, np.ones_like(c+d)], axis=1)
            denominator =  b/denom_first + c/denom_second + 1
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
            nominator = np.square(a*d - b*c)
            denominator = (a+c)*(b+d)*(a+b)*(c+d)
            denominator = np.max(np.c_[denominator, np.ones_like(denominator)], axis=-1)
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
            den_first = np.max(np.c_[(a+c)*(a+b), np.ones_like((a+c)*(a+b))], axis=-1)
            first = a/N*np.log(1+(a*N)/den_first)

            den_second = np.max(np.c_[(b+d)*(a+b), np.ones_like((b+d)*(a+b))], axis=-1)
            second = b/N*np.log(1+(b*N)/den_second)

            den_third = np.max(np.c_[(a+c)*(c+d), np.ones_like((a+c)*(c+d))], axis=-1)
            third = c/N*np.log(1+(c*N)/den_third)

            den_fourth = np.max(np.c_[(b+d)*(c+d), np.ones_like((b+d)*(c+d))], axis=-1)
            fourth = d/N*np.log(1+(d*N)/den_fourth)

            weight_factors.append(first + second + third + fourth)
        return np.max(weight_factors, axis=0) 

    def igm(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        class_based_dfs = np.sort([cat[0] for cat in self.cross_tab], axis=0)[::-1]
        n_classes = class_based_dfs.shape[0]
        max_freq = np.max(class_based_dfs, axis=0)
        igm = max_freq / np.sum(class_based_dfs.T @ np.arange(1, n_classes+1), axis=0)
        return 1+ self.igm_lambda*igm

    def pb(self, X, y):
        if self.cross_tab is None:
            self.make_cross_tab(X, y)

        weight_factors = []
        for category in self.cross_tab:
            a, b, c, d = category
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
        class_factors = np.zeros(shape=(X.shape[1], )) #self.cross_tab[0][0]
        for category in self.cross_tab:
            a, b, c, d = category
            a = a - int(self.smooth_idf)
            class_factors += (a > 0)
        
        icf = np.log((n_classes + int(self.smooth_idf))/(class_factors+int(self.smooth_idf))) + 1
        self.icf_mean = np.mean(icf)
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
            a = a - int(self.smooth_idf)
            D_cls = counts[i]
            class_factors.append(a / D_cls)

        n_classes += int(self.smooth_idf)
        clf_sum = np.sum(class_factors, axis=0) # + int(self.smooth_idf)
        icsdf = np.log(n_classes / clf_sum) + 1
        self.icsdf_mean = np.mean(icsdf)
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