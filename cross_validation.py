#-*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import grid_search
import ipython_parallel as ipp
reload(ipp)
import time


if __name__ == "__main__":


    samples = datasets.load_iris()
    # samples = datasets.load_digits()
    n_samples = len(samples.target)

    clf = svm.SVC(kernel='linear', C=1)

    # cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.1)
    # nested_cv = 10
    cv = cross_validation.LeaveOneOut(n_samples)
    nested_cv = cross_validation.LeaveOneOut(n_samples-1)


    print '#### Cross validation (%s) ####' % cv

    print '## without ipython_cluster (n_jobs=%d)' % -1
    start = time.time()
    scores = cross_validation.cross_val_score(clf, samples.data, samples.target, cv=cv, n_jobs=-1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time

    print '## with ipython_cluster'
    start = time.time()
    scores = ipp.cross_val_score(clf, samples.data, samples.target, cv=cv, n_jobs=-1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time



    print
    print '#### Cross validation (%s) with grid_search (%s) ####' % (cv, nested_cv)
    grids = {'C': [1, 10, 100]}
    clf_grid = grid_search.GridSearchCV(clf, grids, cv=nested_cv)

    print '## without ipython_cluster (n_jobs=%d)' % -1
    start = time.time()
    scores = cross_validation.cross_val_score(clf_grid, samples.data, samples.target, cv=cv, n_jobs=-1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time

    print '## with ipython_cluster'
    start = time.time()
    scores = ipp.cross_val_score(clf_grid, samples.data, samples.target, cv=cv, n_jobs=-1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time

    print '## with ipython_cluster in grid_search'
    clf_grid = ipp.GridSearchCV(clf, grids, cv=nested_cv, grid_parallel=True, n_jobs=4)
    start = time.time()
    scores = cross_validation.cross_val_score(clf_grid, samples.data, samples.target, cv=cv, n_jobs=1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time

    print '## with ipython_cluster in nested cv'
    clf_grid = ipp.GridSearchCV(clf, grids, cv=nested_cv, grid_parallel=False, n_jobs=4)
    start = time.time()
    scores = cross_validation.cross_val_score(clf_grid, samples.data, samples.target, cv=cv, n_jobs=1)
    elapsed_time = time.time() - start
    print '\tCV socres: %f' % scores.mean()
    print '\telapsed_time: %f sec' % elapsed_time
