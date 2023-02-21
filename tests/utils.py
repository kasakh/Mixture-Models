## Utility functions reused across various tests

import autograd.numpy as np

def check_array_equal(actual,expected,exact=False):
    check_dim(actual,np.shape(expected))
    if exact:
        assert np.array_equal(actual,expected)
    else:
        assert np.allclose(actual,expected)

def check_arraydict_equal(actual,expected,**kwargs):
    keys = sorted(actual.keys())
    assert keys == sorted(expected.keys())
    for k in keys:
        check_array_equal(actual[k],expected[k],kwargs)

def check_metric_equal(MM,metric,params_store,expected,**kwargs):
    actual = [getattr(MM,metric)(param) for param in params_store]
    check_array_equal(actual,expected,kwargs)

def check_labels(MM,data,params_store,expected):
    labels = np.array(MM.labels(data,params_store[-1]))
    check_array_equal(labels,expected,exact=True)

def check_init_params(MM,init_params_args,expected,seed):
    np.random.seed(seed)
    init_params = MM.init_params(**init_params_args)
    check_arraydict_equal(init_params,expected)
    return init_params

def init_MM(MMclass,data,seed,init_params_args,expected,**kwargs):
    MM = MMclass(data,**kwargs)
    init_params = check_init_params(MM,init_params_args,expected,seed)
    return MM, init_params

def check_fit(MM,init_params,opt_routine,fit_args,expected_dim,expected_metrics):
    params_store = MM.fit(init_params,opt_routine,**fit_args)
    check_dim(params_store,(expected_dim,))
    for metric in expected_metrics:
        check_metric_equal(MM,metric,params_store,expected_metrics[metric])
    return params_store


"""Functions for testing specific parameters"""

# Verify dimensions of an ndarray
def check_dim(ndarray,expected):
    assert np.shape(ndarray) == tuple(expected)

# Verify evaluation of an ndarray under a given function
## (tolerances can be tweaked with **kwargs)
def check_stat(ndarray, statistic, expected,**kwargs):
    assert np.isclose(statistic(ndarray),expected,**kwargs)

# Verify element of an ndarray
def check_element(ndarray, indices, expected,**kwargs):
    check_stat(ndarray,lambda x:x[tuple(indices)],expected,**kwargs)

# Verify that a matrix is square
def check_square(matrix):
    n = np.shape(matrix)[0]
    check_dim(matrix,(n,n))

# Verify that a matrix is symmetric
def check_symmetric(matrix,**kwargs):
    check_square(matrix)
    assert np.allclose(matrix,np.transpose(matrix),**kwargs)

# Verify that a matrix is diagonal
def check_diagonal(matrix,**kwargs):
    check_square(matrix)
    assert np.allclose(matrix,np.diag(np.diag(matrix)),**kwargs)

# Verify that a matrix is orthogonal
def check_orthogonal(matrix,**kwargs):
    check_square(matrix)
    prod_inv = matrix @ np.linalg.inv(matrix)
    assert np.allclose(prod_inv,np.eye(np.shape(matrix)[0]),**kwargs)

# Verify that a matrix is positive definite (using Cholesky decomposition)
def check_posdef(matrix):
    check_symmetric(matrix)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        assert False

# Verify that all entries of a vector are positive/non-negative
def check_pos(vector,nonneg=True,atol=1e-08):
    mincomp = min(vector)
    if nonneg:
        assert mincomp >= -atol
    else:
        assert mincomp > -atol

# Verify that a matrix is positive semidefinite (by computing eigenvalues)
def check_posdef_eig(matrix,semidef=True,atol=1e-08):
    check_pos(np.linalg.eigvalsh(matrix),semidef,atol)

# Verify that a vector is a probability distribution
def check_probdist(vector,atol=1e-08):
    check_pos(vector,True,atol)
    assert np.isclose(sum(vector),1)
