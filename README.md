# tfidf_based_vectorizers
This repository is devoted to implementation various analogs for TF-IDF vectorizers

Availiable functions:

---------------------

    n_t - Number of times a term occured in the document
    n - number of words in document
    df(t) - document frequency of term t

    idf : log( (1 + n) / (1 + df(t)) ) + 1
    dfs: Distinguishing feature selector
    chi2 : Term Weighting Based on Chi-Square Statistic
    ig: Term weighting based on information gain
    igm: TODO
    pb : Probability-Based Term Weighting
    idf_icf: Term Weighting Based on Inverse Class Frequency
    rf : Term Weighting Based on Relevance Frequency
    idf_icdf: Term Weighting Based on Inverse Class Density Frequency
    iadf : inverse average document frequency
    iadf_norm : inverse average document frequency normalized