import matplotlib.pyplot as plt
import matplotlib.image

import autograd.numpy as np
import autograd.numpy.random as npr
from .data_mnist import *

import csv


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(
    images,
    ax,
    ims_per_row=5,
    padding=5,
    digit_dimensions=(28, 28),
    cmap=matplotlib.cm.binary,
    vmin=None,
    vmax=None,
):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = (N_images - 1) // ims_per_row + 1
    pad_value = np.min(images.ravel())
    concat_images = np.full(
        (
            (digit_dimensions[0] + padding) * N_rows + padding,
            (digit_dimensions[1] + padding) * ims_per_row + padding,
        ),
        pad_value,
    )
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[
            row_start : row_start + digit_dimensions[0],
            col_start : col_start + digit_dimensions[1],
        ] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def make_pinwheel(
    radial_std, tangential_std, num_classes, num_per_class, rate, rs=npr.RandomState(0)
):
    """Based on code by Ryan P. Adams."""
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rs.randn(num_classes * num_per_class, 2) * np.array(
        [radial_std, tangential_std]
    )
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack(
        [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
    )
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum("ti,tij->tj", features, rotations)


default_pinwheel_data = make_pinwheel(0.3, 0.05, 3, 100, 0.4)


def load_csvdataset(file="iris"):
    """Loads example dataset from a CSV file.

    Adapted from `sklearn.datasets._base.load_csv_data`.

    Parameters
    ----------
    file : string
        Name of the dataset to be returned, e.g. "iris", "wine", "Khan_train"

    Returns
    -------
    A tuple of four elements, respectively:

        data
            Matrix of shape (num_datapoints, num_features)
        feature_names
            Optional: Vector of feature names, if present.
        target_names
            Optional: Vector of cluster names, if present.
        truth
            Optional: Ground truth cluster assignments, if present
            (i.e. if the dataset supports supervised training).

    See Also
    --------
    sklearn.datasets._base.load_csv_data

    """
    if file[4:].lower() != ".csv":
        file = file + ".csv"

    with open(
        os.path.join(os.path.join(os.path.dirname(__file__), "datasets"), file)
    ) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        feature_names, target_names = None, None
        if len(temp) > 2:
            feature_names = np.array(temp[2 : (3 + n_features)])
            if len(temp) > 3 + n_features:
                target_names = np.array(temp[(3 + n_features) :])
        data = np.empty((n_samples, n_features))
        truth = np.empty((n_samples,))
        has_ground_truth = None
        for i, ir in enumerate(data_file):
            if has_ground_truth or len(ir) == n_features + 1:
                data[i] = np.asarray(ir[:-1], dtype=np.float64)
                truth[i] = int(ir[-1])
                has_ground_truth = True
            else:
                data[i] = np.asarray(ir, dtype=np.float64)
        return data, feature_names, target_names, truth


def mvn_sample(num_samples, means, sqrt_covs):
    """Samples from the (mixture of) Gaussian distribution.

    Parameters
    ----------
    num_samples : int or vector
        If a vector (arraylike of shape (K,)), interpreted as
        a list of number of observations to sample, for each of K mixture components.
        If an integer, interpreted as a vector of length K=1.

    means : (...,p) vector or list of such vectors
        Mean vector for each of K mixture components.

    sqrt_covs : (...,p,p) matrix or list of such matrices
        Cholesky square root of the covariance matrix, for each of K mixture components.

    Returns
    -------
    A matrix of shape (sum(num_samples),p)
    where the samples from clusters 1, ..., K are concatenated together.
    (The rows are not shuffled.)
    """
    if not len(np.shape(num_samples)):
        num_samples = [num_samples]
        means = [means]
        sqrt_covs = [sqrt_covs]
    num_dim = len(means[0])
    data = [
        np.dot(np.random.randn(num_samples[i], num_dim), sqrt_covs[i]) + means[i]
        for i in range(len(num_samples))
    ]
    return np.concatenate(data)


def simulate_data(
    num_datapoints,
    num_dim,
    num_components,
    constrained,
    balance=1,
    prop_informative=1,
    scale=5,
    seed=None,
):
    """Generates sample data from a Gaussian mixture model, under various constraints.

    Parameters
    ----------
    num_datapoints : int
        Total number of samples to generate (n).

    num_dim : int
        Number of data dimensions (p).

    num_components : int
        Number of clusters (K).

    constrained : bool
        If true, the covariance matrix is constrained to be diagonal.
        If false,

    balance : vector or float, optional
        Either a probability distribution (vector of length K)
        indicating how the n are to be distributed among the clusters,
        or a single (positive) float
        indicating the ratio of number of datapoints in the first cluster
        to the average number of datapoints in the other K-1 clusters.
        (Due to rounding, this ratio may be merely approximate.)
        Defaults to 1 (= balanced dataset).

    prop_informative : float, optional
        A single float between 0 and 1 inclusive
        such that only the first max(1,prop_informative*num_dim) features are informative
        in the sense that the remaining features are replaced with random noise.
        Defaults to 1 (= all features informative).

    Returns
    -------
    data : (num_datapoints, num_dim) matrix
        The randomly generated data matrix.
    num_samples : (num_components,) vector
        Number of sample observations for each component.
    means : (num_components, num_dim) matrix
        Centers for each component.
    covariances : (num_components, num_dim, num_dim) array
        Covariance matrices for each component.

    Other Parameters
    ----------------
    scale : float, optional
        Scale parameter for the mean (feature) values. Defaults to 5.

    seed : int64, optional
        Set the seed, if supplied.
         Defaults to None ( = unset).

    See Also
    --------
    mvn_sample : Invokes Gaussian distribution sampler.
    numpy.random.seed
    """
    if seed:
        npr.seed(seed)

    # Constructs the vector of ground truths
    if not len(np.shape(balance)):
        balance = [int(balance * (num_components - 1))] + [1] * (num_components - 1)
        balance = [p / sum(balance) for p in balance]
    num_samples = [int(prop * num_datapoints / num_components) for prop in balance[1:]]
    num_samples = [num_datapoints - sum(num_samples)] + num_samples

    # Construct an array of centers, of shape (K,p)
    num_informative_features = max(1, int(num_dim * prop_informative))
    means = np.random.rand(num_components, num_informative_features) * scale
    means = np.array(sorted(means.tolist(), key=lambda m: sum(x * x for x in m)))
    means = np.concatenate(
        [
            means.transpose(),
            np.zeros((num_dim - num_informative_features, num_components)),
        ]
    ).transpose()  # Pad uninformative features with 0s

    # Constructs the (square roots of the) covariance matrices
    if constrained:
        sqrt_covs = np.repeat(
            [np.diag(np.random.rand(num_dim))], num_components, axis=0
        )
    else:
        sqrt_covs = np.random.randn(num_components, num_dim, num_dim) / np.sqrt(num_dim)

    covariances = np.array([np.dot(x, x.transpose()) for x in sqrt_covs])

    return mvn_sample(num_samples, means, sqrt_covs), num_samples, means, covariances
