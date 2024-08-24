# Mixture-Models

[![codecov](https://codecov.io/gh/kasakh/Mixture-Models/graph/badge.svg?token=znrlf0JRsD)](https://codecov.io/gh/kasakh/Mixture-Models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A one-stop Python library for fitting a wide range of mixture models such as Mixture of Gaussians, Students'-T, Factor-Analyzers, Parsimonious Gaussians, MCLUST, etc. 


## Table of Contents
- [Why this library](#why-this-library)
- [Installation and Quick Start](#installation-and-quick-start)
- [Supported Models and Optimization Routines](#supported-models-and-optimization-routines)
- [Contributing](#contributing)
- [Citating this library](#citation)

## Why this library
While there are several packages in R and Python which support various kinds of mixture-models, each one of them has their own API and syntax. Further, in almost all those libraries, the inference proceeds via Expectation-Maximization (a Quasi first order method) which makes them unsuitable for high-dimensional data. 

This library attempts to provide a seamless and unified interface for fitting a wide-range of mixture models. Unlike many existing packages that rely on Expectation-Maximization for inference, our approach leverages Automatic Differentiation tools and gradient-based optimization which makes it well equipped to handle high-dimensional data and second order optimization routines. 

## Installation and Quick Start

Installation is straightforward:

	pip install Mixture-Models

#### Quick Start

```python
from mixture_models import *
```

The estimation procedure consists of 3 simple steps:

```python
    ### Simulate some dummy data using the built-in function make_pinwheel
    data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                        num_per_class=100, rate=0.4,rs=npr.RandomState(0))

    ### Plot the three clusters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:, 0], data[:, 1], 'k.')
    plt.show()

    ### STEP 1 - Choose a mixture model to fit on your data
    my_model = GMM(data)

    ### STEP 2 - Initialize your model with some parameters    
    init_params = my_model.init_params(num_components = 3,scale = 0.5)

    ### STEP 3 - Learn the parameters using some optimization routine
    params_store = my_model.fit(init_params,"Newton-CG")

```

Once the model is trained on the data (which is a `numpy` matrix of shape `(num_datapoints, num_dim)`),
post-hoc analysis can be performed:

```python
    for params in params_store:
        print("likelihood",my_model.likelihood(params))
        print("aic,bic",my_model.aic(params),my_model.bic(params))
    
    np.array(my_model.labels(data,params_store[-1])) ## final predicted labels
```

Example notebooks are available on the [project Github repo](https://github.com/kasakh/Mixture-Models).

## Supported models and optimization routines

There are more than 30+ different mixture-models, spread across five model families, currently supported by the library. Here is a brief overview of the different model families supported:

- `GMM`: Standard *Gaussian mixture model*
    - `GMM_Constrainted`: GMM with common covariance across components
    - `Mclust`: [MCLUST family](https://sites.stat.washington.edu/raftery/Research/PDF/fraley2003.pdf) of constrained GMMs
- `MFA`: [*Mixture-of-factor analyzers*](https://link.springer.com/article/10.1007/s11222-008-9056-0)
    - `PGMM`: [Parsimonious GMM extension](https://link.springer.com/article/10.1007/s11222-008-9056-0) with constraints
- `TMM`: [*Mixture of t-distributions*](https://www.academia.edu/16834403/Robust_mixture_modelling_using_the_t_distribution?sm=b)

The [project repo 'Examples' folder](https://github.com/kasakh/Mixture-Models/tree/master/Mixture_Models) includes more detailed illustrations for all these models, as well as a `README.md` for advanced users who want to fit custom mixture models,
or tinker with the settings for the above procedure.

Currently, four main gradient based optimizers are available:

- `"grad_descent"`: Stochastic Gradient Descent (SGD) with momentum
- `"rms_prop"`: Root-mean-squared propagation (RMS-Prop)
- `"adam"`: Adaptive moments (ADAM)
- `"Newton-CG"`: Newton-Conjugate Gradient (Newton CG)

The details about each optimizer and its optional input parameters are given in the PDF in the 'Examples' folder.
The output of `fit` method is the set of all points in the parameter space that the optimizer has traversed during the optimization i.e.  list of parameters with the final entry in the list being the final fitted solution.
We have a detailed notebook `Optimizers_illustration.ipynb` in the 'Examples' folder on Github. 

## Contributing
We welcome contributions to our library. Our code base is highly modularized, making it easy for new contributors to extend its capabilities and add support for additional models. If you are interested in contributing to the library, check out the [contribution guide](contributing.md).

If you're unsure where to start, check out our open issues for inspiration on the kind of problems you can work on.  Alternately, you could also open a new issue so we can discuss the best strategy for integrating your work.


-------------------------------------------------------------------------------

If you use this package, please consider citing our research as 


 <blockquote>
        <p>@article{kasa2024mixture,
  title={Mixture-Models: a one-stop Python Library for Model-based Clustering using various Mixture Models},
  author={Kasa, Siva Rajesh and Yijie, Hu and Kasa, Santhosh Kumar and Rajan, Vaibhav},
  journal={arXiv preprint arXiv:2402.10229},
  year={2024}
}
}</p>
    </blockquote>

 <blockquote>
        <p>@article{kasa2020model,
  title={Model-based Clustering using Automatic Differentiation: Confronting Misspecification and High-Dimensional Data},
  author={Kasa, Siva Rajesh and Rajan, Vaibhav},
  journal={arXiv preprint arXiv:2007.12786},
  year={2020}
}</p>
    </blockquote>


