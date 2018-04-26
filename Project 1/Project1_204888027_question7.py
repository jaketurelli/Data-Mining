if __name__ == '__main__':
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import NMF
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from nltk import pos_tag
    import nltk
    import pandas as pd



    categories = ['comp.graphics', 'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey']
    dataset_trn_01 = fetch_20newsgroups(subset='train',categories=categories,
                                       shuffle=True, random_state=42)
    dataset_tst_01 = fetch_20newsgroups(subset='test',categories=categories,
                                       shuffle=True, random_state=42)
    dataset_trn_02 = fetch_20newsgroups(subset='train',categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=('headers','footers'))
    dataset_tst_02 = fetch_20newsgroups(subset='test',categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=('headers','footers'))

    print(np.unique(np.array(dataset_trn_01.target)))
    print(np.unique(np.array(dataset_tst_01.target)))
    # combine documents
    i = 0
    for t in dataset_trn_01.target:
        if t < 4:
            dataset_trn_01.target[i] = 0
        else:
            dataset_trn_01.target[i] = 1
        i=i+1
    i = 0
    for t in dataset_tst_01.target:
        if t < 4:
            dataset_tst_01.target[i] = 0
        else:
            dataset_tst_01.target[i] = 1
        i=i+1

    i = 0
    for t in dataset_trn_02.target:
        if t < 4:
            dataset_trn_02.target[i] = 0
        else:
            dataset_trn_02.target[i] = 1
        i=i+1
    i = 0
    for t in dataset_tst_02.target:
        if t < 4:
            dataset_tst_02.target[i] = 0
        else:
            dataset_tst_02.target[i] = 1
        i=i+1
    print('--')
    print(np.unique(np.array(dataset_trn_01.target)))
    print(np.unique(np.array(dataset_tst_01.target)))



    wnl = nltk.wordnet.WordNetLemmatizer()
    def penn2morphy(penntag):
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN':'n', 'JJ':'a',
                      'VB':'v', 'RB':'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n'

    def lemmatize_sent(list_word):
        # Text input is string, returns array of lowercased strings(words).
        return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
                for word, tag in pos_tag(list_word)]
    analyzer = CountVectorizer().build_analyzer()
    def stem_rmv_punc(doc):
        return (word for word in lemmatize_sent(analyzer(doc)) if not word.isdigit())
        #return (word for word in lemmatize_sent(analyzer(doc)) if word not in stop_words_english and not word.isdigit())


    MIN_DF_OPTIONS = [3, 5]
    strength_L1 = 10
    strength_L2 = 100
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', TruncatedSVD(n_components=50, random_state=42)),
        ('clf', GaussianNB()),
    ])
    print(pipeline)
    param_grid = [
        {
            'vect': [CountVectorizer(stop_words='english',analyzer=stem_rmv_punc),
                     CountVectorizer(stop_words='english')],
            'vect__min_df': MIN_DF_OPTIONS,
            'tfidf': [TfidfTransformer()],
            'reduce_dim': [TruncatedSVD(n_components=50), NMF(n_components=50, init='random', random_state=42)],
            'clf': [svm.SVC(C=10, kernel='linear', random_state = 15, probability = True),
                    LogisticRegression(penalty = 'l1', C=1/strength_L1),
                    LogisticRegression(penalty = 'l2', C=1/strength_L2),
                    GaussianNB()]
        }]

    # used to cache results
    from tempfile import mkdtemp
    from shutil import rmtree
    from sklearn.externals.joblib import Memory
    # print(__doc__)
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=10)
    grid = GridSearchCV(pipeline, cv=5, n_jobs=3, param_grid=param_grid, scoring='accuracy',verbose = 1,refit=False,return_train_score=True)
    print(grid)
    grid.fit(dataset_trn_01.data, dataset_trn_01.target)
    myPD1=pd.DataFrame(grid.cv_results_)
    print(myPD1)

    grid.fit(dataset_trn_02.data, dataset_trn_02.target)
    myPD2=pd.DataFrame(grid.cv_results_)
    print(myPD2)

    rmtree(cachedir)