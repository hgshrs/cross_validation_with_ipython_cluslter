A library for cross_val_score() and GridSearchCV() (in scikit-learn) with IPython cluster.

For the usage, see cross_validation.py.

For the details of IPython cluster, you can refer http://ipython.org/ipython-doc/dev/parallel/ and http://qiita.com/chokkan/items/750cc12fb19314636eb7 (in Japanese).

```python
In [1]: run cross_validation
#### Cross validation (sklearn.cross_validation.LeaveOneOut(n=150)) ####
## without ipython_cluster (n_jobs=-1)
	CV socres: 0.980000
	elapsed_time: 0.338331 sec
## with ipython_cluster
	CV socres: 0.980000
	elapsed_time: 0.116183 sec

#### Cross validation (sklearn.cross_validation.LeaveOneOut(n=150)) with grid_search (sklearn.cross_validation.LeaveOneOut(n=149)) ####
## without ipython_cluster (n_jobs=-1)
	CV socres: 0.980000
	elapsed_time: 5.717641 sec
## with ipython_cluster
	CV socres: 0.980000
	elapsed_time: 2.754487 sec
## with ipython_cluster in grid_search
	CV socres: 0.980000
	elapsed_time: 5.590090 sec
## with ipython_cluster in nested cv
	CV socres: 0.980000
	elapsed_time: 5.570679 sec
```
