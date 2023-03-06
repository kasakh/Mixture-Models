'''
Utility functions for checking specific parameter values, also used for testing
'''

import autograd.numpy as np

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

# Verify that an ndarray contains only finite entries (e.g. no NANs or Infs)
def check_finite(ndarray,**kwargs):
    assert np.all(np.isfinite(ndarray,**kwargs))

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

# Verify that a matrix is nonsingular
def check_invertible(matrix,**kwargs):
    check_square(matrix)
    assert np.linalg.cond(matrix) < 1/kwargs.get('atol',1e-08)

# Verify that a matrix is orthogonal
def check_orthogonal(matrix,**kwargs):
    check_invertible(matrix,**kwargs)
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
    check_finite(vector)
    if nonneg:
        assert np.all(vector >= -atol)
    else:
        assert np.all(vector > -atol)

# Verify that a matrix is positive semidefinite (by computing eigenvalues)
def check_posdef_eig(matrix,semidef=True,atol=1e-08):
    check_pos(np.linalg.eigvalsh(matrix),semidef,atol)

# Verify that a vector is a probability distribution
def check_probdist(vector,atol=1e-04):
    check_pos(vector,True)
    assert np.isclose(sum(vector),1,atol)