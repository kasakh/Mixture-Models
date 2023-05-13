# Advanced Usage

This is a detailed guide for users who want to write their own mixture model classes,
or at least looking to understand how to better utilize the interface offered by this package.

We will use the `Mclust` mixture model family as a running example.

## Defining a Custom Mixture Model Class

The base class used by this package is `MM` (for "Mixture Model"), defined in `Mixture_Models\mixture_models.py`.
Note that the same folder also contains modules for each of the custom subclasses.
In particular, `Mclust` is located in `mclust.py`.

As illustrated by the quick start example, the bare minimum functionality required of a mixture model class includes:

1. A *constructor* `__init__` that takes input data and instantiates the model;
2. A method `init_params` for *initializing the model parameters*;
3. A method `fit` for *training the model to learn the parameters from data*.

In practice, you will also want methods to support post-hoc analysis,
including at least a method `labels` for predicting the latent component of a given observation (or dataset of observations).

You can confirm that `Mclust` -- and indeed, all of the other model classes currently implemented in this package -- have all of these methods.

## Constructor

The base `MM` class provides a simple constructor `__init__(self, data)`
that simply performs some checks for missing/non-finite values in the data.
For some of the models (such as the simple Gaussian mixture model `GMM`),
this constructor can be inherited directly,
hence there is no need to write a separate `__init__` routine.
However, other models may require additional specifications
that have to be supplied upon instantiation.
For example, `Mclust` comprises a [family of mixture models](https://sites.stat.washington.edu/raftery/Research/PDF/fraley2003.pdf) whose component covariances are specified as eigendecompositions $\Sigma_k = \lambda_k D_k A_k D_k^\top $.
This imposes various constraints, encoded as three-letter strings like `"EEV"` and `"VVI"`, indicating which of these parameters are shared across components. This constraint must be specified upon model instantiation, and saved along with the data. In `Mclust`, this is done by extending the constructor to take an additional argument `constraint`, which will be used for the next step.

## Model parameters and Reparametrization

Gradient descent is a powerful and generic optimization routine, that can be used to learn the maximum likelihood estimates for many classes of statistical models (for example, deep neural networks). However, for mixture models, there is a technical obstacle:

> Mixture model parameters are subject to constraints (if nothing else, that the mixture weights are positive and sum to $1$), while gradient-based method assume an unconstrained optimization.

The good news is that this obstacle can be surmounted, by *reparametrizing the model* so that the objective (log-likelihood) becomes an unconstrained function of its transformed parameters. The bad news is that this reparametrization must be done manually for each class that we want to implement, and may require some creativity on our part.

Here are some examples:

- The mixture weights $[p_1, \dots, p_K] > 0$, $p_1 + \dots + p_K=1$ can be converted to unconstrained parameters $p_1' = \log p_1, \dots, p_K' = \log p_K$, which are then normalized using the *log-sum-exp trick*.
- For the simple GMM, the constraint that the component covariance matrices $\Sigma_k$ have to be positive definite can be removed by *Cholesky decomposition* $\Sigma_k=(\Sigma_k')(\Sigma_k')^\top$.
- For the `Mclust` family, the constraint that the eigenvector matrices $D_k$ have to be orthogonal can be removed by *Cayley transformation* $D_k=(I+S_k)(I-S_k)^{-1}$, where $S_k$ is skew-symmetric (and hence equal to $(D_k'-(D_k')^\top)/2$ for some unconstrained matrix $D_k'$).

More details can be found in section 3 of [this paper](https://arxiv.org/pdf/2007.12786.pdf).

Assuming that you have derived a suitable reparametrization for your desired mixture model class,
you will want to implement the following methods:

- `init_params`: Instantiates the transformed (unconstrained) parameters for the model.
In particular, the first argument should be `num_components`, the number of clusters to fit.
    - Typically, you will also want a `scale` parameter, to control the size of the initialized means and covariances (this plays an important role if the data is not normalized).
    - Many of the implemented models take `use_kmeans` as a flag
    that initializes the parameters using the *K-Means algorithm*.
    However, this is not strictly necessary, and for the estimation routine the user can manually replace the initialization with another one of their own specification.
- `unpack_params`: Implements the (inverse) derived reparametrization.
Essentially, it takes as input the transformed parameters (for example, as obtained from `init_params`)
and returns the original (constrained) parameters that would enter into the model log-likelihood function.
    - The function body of `unpack_params` and `init_params` will generally depend on attributes like `"constraint"` that are set at model instantiation.
    - To support post-hoc analysis, e.g. calculation of AIC/BIC,
    we also recommend that you define an attribute `num_freeparams`
    that computes the number of degrees of freedom for each (constrained) model instance.
- `objective`: The (negative) log-likelihood function, that is to be optimized via automatic differentiation.
    - As an example, for both `GMM` and `Mclust` this function is the multivariate normal probability distribution, evaluated on the input data.
- `fit`: Performs automatic differentiation on the function `objective`,
hence learning the (transformed) parameters from the data.
    - This function is currently re-implemented in each mixture model subclass
    in order to support the auxiliary method `params_checker`,
    which is run on each iteration to check if
    the current parameter values has hit a singularity.
    The rest of the function body is essentially identical across subclasses
    and can probably be copied over directly.
    - In particular, the optimization routine to be used is indicated by the argument `"opt_routine"`
    which is passed directly to the base method `MM.optimize`
    where the actual gradient optimization is performed.

## Writing Tests

Once all these methods have been implemented for your custom mixture model class,
you should be able to run the 3-step estimation procedure as outlined in the quick start.
If you want to verify the consistency of your implementation,
you can write a simple test suite using the helper functions in `tests\utils.py`.
The general pattern for writing a functional test can be thought of as an extension of the 3-step estimation routine:

    Step 0a: Set the random seed (for reproducibility)
    Step 0b: Import or simulate the data to be used.
    Step 1-3: Run the 3-step estimation routine, instantiating your model and fitting it to the data.
    Step 4: Perform post-hoc analysis, if any.
    Step 5: Verify that the fitted parameter estimates (and post-hoc analysis results) agree with the expected values.

In this package, each model has its own tests located in a module in the `tests` folder (e.g. `tests\gmm.py`, `tests\mclust.py`), which you may refer to for more details.
Additionally, the code that implements the functional test pattern described above
has also been collected into a Jupyter notebook in the `Examples` folder (found under `GMM_illustration.ipynb`, `Mclust_illustration.ipynb` etc.)
