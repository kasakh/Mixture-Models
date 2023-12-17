"""Utility functions for checking specific parameter values.

These functions are primarily used to verify
that an input parametrization has the correct dimensions and ranges.
They are also used in testing, but have been moved into userspace.

"""

import autograd.numpy as np


def check_dim(ndarray, expected):
    """Verifies that a given ndarray has the expected dimensions.

    Parameters
    ----------
    ndarray
        Input to be tested.
    expected : array_like
        Expected dimensions of `ndarray.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.arange(24).reshape((2,3,4))
    >>> check_dim(x,(2,3,4))
    >>> check_dim(x,[2,3,4])
    >>> check_dim(x[1],(3,4))
    >>> check_dim(x[1][2][3],())
    """
    assert np.shape(ndarray) == tuple(expected)


def check_stat(ndarray, statistic, expected, **kwargs):
    """Verifies evaluation of an ndarray under a given unary function.

    Parameters
    ----------
    ndarray
        Argument of the evaluation.
    statistic : function
        Function that evaluates `ndarray` and returns a result. Must be unary.
    expected
        Expected result of the function call `statistic(ndarray)`.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the value comparison.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isclose

    Examples
    --------
    >>> x = [1/3, 1/2, 1/6]
    >>> check_stat(x, sum, 1)
    >>> y = [np.log(i) for i in x]
    >>> check_stat(y, lambda x: sum(np.exp(x)), 1, atol=1e-04)
    """
    assert np.isclose(statistic(ndarray), expected, **kwargs)


def check_element(ndarray, indices, expected, **kwargs):
    """Verify that an element of an ndarray has an expected value.

    Parameters
    ----------
    ndarray
        Argument of the evaluation.
    indices : array_like
        Indices of the element to be tested.
    expected
        Expected value of the element.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the value comparison.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isclose

    Examples
    --------
    >>> x = np.arange(24).reshape((2,3,4))/24
    >>> check_element(x, (1,2,1), 0.875)
    >>> check_element(x, [0,0,2], 0.08333, atol=1e-04)
    """
    check_stat(ndarray, lambda x: x[tuple(indices)], expected, **kwargs)


def check_finite(ndarray, **kwargs):
    """Verifies that an ndarray contains only finite entries.

    In particular, it verifies that none of the entries are NANs or +/-Inf.

    Parameters
    ----------
    ndarray
        Input to be tested.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isfinite` call).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isfinite

    Examples
    --------
    >>> x = np.array([1/2, 0])
    >>> check_finite(x)
    >>> # check_finite(np.log(x)) ##throws an error
    >>> x = x - 0.00000001
    >>> check_finite(x)
    >>> # check_finite(np.log(x)) ##throws an error
    """
    assert np.all(np.isfinite(ndarray, **kwargs))


def check_square(matrix):
    """Verifies that a given matrix is square.

    Parameters
    ----------
    matrix
        Input to be tested.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_square(x)
    >>> # check_square(x.flatten()) #also throws an error for nonmatrices
    """
    n = np.shape(matrix)[0]
    check_dim(matrix, (n, n))


def check_symmetric(matrix, **kwargs):
    """Verifies that a given matrix is symmetric.

    Parameters
    ----------
    matrix
        Input to be tested.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the element comparison.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isclose

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_symmetric(x)
    >>> # check_symmetric(x.flatten()) #also throws an error for nonmatrices
    >>> # check_symmetric(x.reshape((2,8))) #also throws an error for nonsquare matrices
    """
    check_square(matrix)
    assert np.allclose(matrix, np.transpose(matrix), **kwargs)


def check_diagonal(matrix, **kwargs):
    """Verifies that a given matrix is diagonal.

    Parameters
    ----------
    matrix
        Input to be tested.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the element comparison.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isclose

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_diagonal(x)
    >>> # check_diagonal(x.flatten()) #also throws an error for nonmatrices
    >>> # check_diagonal(x.reshape((2,8))) #also throws an error for nonsquare matrices
    """
    check_square(matrix)
    assert np.allclose(matrix, np.diag(np.diag(matrix)), **kwargs)


def check_invertible(matrix, **kwargs):
    """Verifies that a given matrix is nonsingular.

    This attempts to compute the condition number of the matrix,
    and checks that it is sufficiently small (i.e. < 1e+08,
    this threshold is adjustable by passing in a keyword argument 'atol').

    Parameters
    ----------
    matrix
        Input to be tested.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the condition number comparison.
        Currently only the "atol" keyword is used.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_invertible(x)
    >>> # check_invertible(x.flatten()) #also throws an error for nonmatrices
    >>> # check_invertible(x.reshape((2,8))) #also throws an error for nonsquare matrices
    >>> y = np.ones((4,4)) - x*1e-08
    >>> # check_invertible(y) #insufficiently conditioned
    >>> check_invertible(y, atol=1e-09)
    """
    check_square(matrix)
    assert np.linalg.cond(matrix) < 1 / kwargs.get("atol", 1e-08)


def check_orthogonal(matrix, **kwargs):
    """Verifies that a given matrix is orthogonal.

    Parameters
    ----------
    matrix
        Input to be tested.
    **kwargs : dict, optional
        Other parameters (passed to `numpy.isclose` call)
        for adjusting tolerance of the element comparison.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.isclose

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_orthogonal(x)
    >>> # check_orthogonal(x.flatten()) #also throws an error for nonmatrices
    >>> # check_orthogonal(x.reshape((2,8))) #also throws an error for nonsquare matrices
    >>> # check_orthogonal(np.ones((4,4))) #also throws an error for noninvertible matrices
    """
    check_invertible(matrix, **kwargs)
    prod_inv = matrix @ np.linalg.inv(matrix)
    assert np.allclose(prod_inv, np.eye(np.shape(matrix)[0]), **kwargs)


def check_posdef(matrix):
    """Verifies that a matrix is positive definite (via Cholesky decomposition).

    This test simply calls the routine `numpy.linalg.cholesky` on its input
    and reports the success of the result.

    Parameters
    ----------
    matrix
        Input to be tested.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    See Also
    --------
    numpy.linalg.cholesky

    Examples
    --------
    >>> x = np.eye(4)
    >>> check_posdef(x)
    """
    check_symmetric(matrix)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        assert False


def check_pos(vector, nonneg=True, atol=1e-08):
    """Verifies that all entries of a vector are positive/non-negative.

    Parameters
    ----------
    vector
        Input to be tested.
    nonneg : bool, optional
        Flag indicating whether to test for nonnegativity (True)
        or strict positivity (False). Defaults to True.
    atol : float, optional
        Small positive number indicating the allowed absolute tolerance.
        Defaults to 1e-08.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.array([10**(-i) for i in range(5)])
    >>> check_pos(x)
    >>> check_pos(x, nonneg=False)
    >>> y = x*1e-04 - 2e-08
    >>> check_pos(y)
    >>> check_pos(y, nonneg=False, atol=1e-07)
    """
    check_finite(vector)
    if nonneg:
        assert np.all(vector >= -atol)
    else:
        assert np.all(vector > -atol)


def check_posdef_eig(matrix, semidef=True, atol=1e-08):
    """Verifies that a matrix is positive definite/semidefinite (by computing eigenvalues).

    Parameters
    ----------
    matrix
        Input to be tested.
    semidef : bool, optional
        Flag indicating whether to test for positive semidefiniteness (True)
        or strict positive definiteness (False). Defaults to True.
    atol : float, optional
        Small positive number indicating the allowed absolute tolerance
        for the largest eigenvalue. Defaults to 1e-08.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.array([10**(-i) for i in range(5)])
    >>> check_pos(np.diag(x))
    >>> check_pos(np.diag(x), nonneg=False)
    >>> y = x*1e-04 - 2e-08
    >>> check_pos(np.diag(y))
    >>> check_pos(np.diag(y), nonneg=False, atol=1e-07)
    """
    check_pos(np.linalg.eigvalsh(matrix), semidef, atol)


def check_probdist(vector, atol=1e-04):
    """Verifies that a vector is a probability distribution.

    Parameters
    ----------
    vector
        Input to be tested.
    atol : float, optional
        Small positive number indicating the allowed absolute tolerance
        for the sum of entries from 1. Defaults to 1e-08.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Upon test failure.

    Examples
    --------
    >>> x = np.array([1/3, 1/2, 0, 1/6])
    >>> check_probdist(x)
    """
    check_pos(vector, True)
    assert np.isclose(sum(vector), 1, atol=atol)
