#-*- coding: utf-8 -*-
from IPython.parallel import Client
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.base import clone
import numpy as np

def cross_val_score(estimator, X, y, scoring=None, cv=10, profile='net', n_jobs=-1, verbose=None):
    input_sets = []
    if type(cv) == int:
        cv = cross_validation.KFold(len(X), n_folds=cv)
    for train, test in cv:
        input_sets.append({
                'X_train':X[train],
                'X_test':X[test],
                'y_train':y[train],
                'y_test':y[test],
                'estimator':estimator,
                'scoring':scoring
                })
    rc = Client(profile=profile)
    if n_jobs == -1:
        dview = rc[:]
    else:
        dview = rc[:n_jobs]
    results = dview.map_sync(score_out, input_sets)
    return np.array(results)

def grid_cv_scores(estimator, X, y, grid, scoring=None, cv=10, profile='net', n_jobs=-1, verbose=None):
    input_sets = []
    if type(cv) == int:
        cv = cross_validation.KFold(len(X), n_folds=cv)
    for parameters in grid:
        estimator1 = clone(estimator)
        input_sets.append({
            'estimator':estimator1.set_params(**parameters),
            'X':X,
            'y':y,
            'scoring':scoring,
            'cv':cv
            })
    rc = Client(profile=profile)
    if n_jobs == -1:
        dview = rc[:]
    else:
        dview = rc[:n_jobs]
    results = dview.map_sync(scores_out, input_sets)
    scores = []
    grid_scores = []
    for ii in range(len(grid)):
        scores.append(results[ii].mean())
        grid_scores.append(grid_search._CVScoreTuple(
            list(grid)[ii],
            results[ii].mean(),
            results[ii]))
    return scores, grid_scores

def scores_out(input_sets):
    from sklearn import cross_validation
    estimator = input_sets['estimator']
    X = input_sets['X']
    y = input_sets['y']
    scoring = input_sets['scoring']
    cv = input_sets['cv']
    return cross_validation.cross_val_score(estimator, X, y, scoring, cv)

def score_out(input_set):
    # import socket
    # print socket.getfqdn()
    estimator = input_set['estimator']
    estimator.fit(input_set['X_train'], input_set['y_train'])
    if input_set['scoring']:
        y_pred = estimator.predict(input_set['X_test'])
        return input_set['scoring'](input_set['y_test'], y_pred)
    else:
        return estimator.score(input_set['X_test'], input_set['y_test'])

class GridSearchCV(grid_search.BaseSearchCV):

    def __init__(self, estimator, param_grid, profile=None, grid_parallel=True, scoring=None, loss_func=None,
            score_func=None, fit_params=None, n_jobs=1, iid=True,
            refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
            error_score='raise'):

        super(GridSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch)
        self.param_grid = param_grid
        self.profile = profile
        self.grid_parallel = grid_parallel
        grid_search._check_param_grid(param_grid)


    def fit(self, X, y=None):
        if self.profile:
            return self.fit_ipp(X, y, grid_search.ParameterGrid(self.param_grid))
        else:
            return self._fit(X, y, grid_search.ParameterGrid(self.param_grid))

    def fit_ipp(self, X, y, grid):
        if self.grid_parallel:
            scores, grid_scores = grid_cv_scores(self.estimator, X, y, grid, self.scoring, self.cv,
                    self.profile, self.n_jobs, self.verbose)
        else:
            scores = []
            # grid =  grid_search.ParameterGrid(self.param_grid)
            grid_scores = [];
            for parameters in grid:
                self.estimator.set_params(**parameters)
                scores_cv = cross_val_score(self.estimator, X, y, self.scoring, self.cv, profile=self.profile)
                scores.append(np.array(scores_cv).mean())
                grid_scores.append(grid_search._CVScoreTuple(
                    parameters,
                    scores_cv.mean(),
                    scores_cv))

        max_idx = np.array(scores).argmax()
        self.best_estimator_ = self.estimator.set_params(**list(grid)[max_idx])
        self.best_params_ = list(grid)[max_idx]
        self.scores_ = scores
        self.best_score_ = np.array(scores).max()
        self.grid_scores_ = grid_scores

        if self.refit:
            self.best_estimator_.fit(X, y)

        return self


if __name__ == "__main__":

    rc = Client(profile='net')
    A = rc.queue_status()
    for ii in range(len(rc)):
        print A[ii]
