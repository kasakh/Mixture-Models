"""Utility functions reused across various tests."""

from mixture_models import checkers
import autograd.numpy as np
import json
import os
import math


def check_two_scalars_are_within_bounds(a, b, rel_bound):
    return math.isclose(a, b, rel_tol=rel_bound)


def check_array_equal(actual, expected, atol=1e-04):
    """Verifies that an ndarray has the expected dimension and values."""
    checkers.check_dim(actual, np.shape(expected))
    if atol is None:
        assert np.array_equal(actual, expected)
    else:
        assert np.allclose(actual, expected, atol=atol)


def check_last_element_array_dict_equal(actual, expected, **kwargs):
    keys = sorted(actual.keys())
    assert keys == sorted(expected.keys())

    for k in keys:
        check_two_scalars_are_within_bounds(
            actual[k][-1], expected[k][-1], rel_bound=1e-2
        )


def check_arraydict_equal(actual, expected, **kwargs):
    """Verifies that a dictionary of ndarrays has the expected names and contents."""
    keys = sorted(actual.keys())
    assert keys == sorted(expected.keys())

    for k in keys:
        check_array_equal(actual[k], expected[k], **kwargs)


def check_metric_equal(MM, metric, params_store, expected, **kwargs):
    """Verifies evaluation of a list of parameters under a given mixture model method (aka metric)."""
    actual = [getattr(MM, metric)(param) for param in params_store]
    check_array_equal(actual, expected, **kwargs)


def check_labels(MM, data, params_store, expected):
    """Verifies final assignment of data for a given history of fitted mixture model parameters."""
    labels = np.array(MM.labels(data, params_store[-1]))
    check_array_equal(labels, expected, atol=None)


def init_MM(MMclass, data, seed, init_params_args, expected, **kwargs):
    """Initializes a mixture model and verifies that it has the expected values."""
    MM = MMclass(data, **kwargs)
    np.random.seed(seed)
    init_params = MM.init_params(**init_params_args)
    check_arraydict_equal(init_params, expected)
    return MM, init_params


def check_fit(MM, init_params, opt_routine, fit_args, expected_dim, expected_metrics):
    """Fits a mixture model and verifies that it has the expected values under given evaluation metrics."""
    params_store = MM.fit(init_params, opt_routine, **fit_args)
    checkers.check_dim(params_store, (expected_dim,))
    for metric in expected_metrics:
        check_metric_equal(MM, metric, params_store, expected_metrics[metric])
    return params_store


def check_fromfile(MM, init_params, opt_routine, fit_args, filepath, metrics):
    """Fits a mixture model and verifies the results against expected values imported from a file."""
    params_store = MM.fit(init_params, opt_routine, **fit_args)
    actual_metrics = {
        metric: [getattr(MM, metric)(param) for param in params_store]
        for metric in metrics
    }
    actual_labels = np.array(MM.labels(MM.data, params_store[-1]))

    filepath = os.path.join("tests", filepath)
    if not os.path.isfile(filepath):  # create the file
        create_run([params_store, actual_metrics, actual_labels], filepath)

    with open(filepath) as param_file:
        expected_params_store, expected_metrics, expected_labels = json.load(param_file)

    check_two_scalars_are_within_bounds(
        len(params_store), len(expected_params_store), rel_bound=1e-2
    )

    check_arraydict_equal(
        params_store[-1], expected_params_store[-1], atol=1e-2
    )  # checking if the final convergence value is close to expected

    check_last_element_array_dict_equal(actual_metrics, expected_metrics, atol=1e-2)

    check_array_equal(actual_labels, expected_labels, atol=1e-1)
    return params_store


def create_run(run, filepath):
    # Numpy serializer code, written by 'karlB' and retrieved from https://stackoverflow.com/a/47626762
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(filepath, "w") as param_file:
        json.dump(run, param_file, cls=NumpyEncoder)
