# Mixture-Models

A Python library for fitting mixture models using gradient based inference.
Example notebooks are available on the [project Github repo](https://github.com/kasakh/Mixture-Models).

Installation is straightforward:

	pip install Mixture-Models

## Quick Start

The estimation procedure consists of 3 simple steps:

    ### Choose a mixture model to fit on your data
    my_model = GMM(data)

    ### Initialize your model with some parameters    
    init_params = my_model.init_params(num_components = 3,scale = 0.5)

    ### Learn the parameters using some optimization routine
    params_store = my_model.fit(init_params,"Newton-CG")

Once the model is trained on the data (which is a `numpy` matrix of shape `(num_datapoints, num_dim)`),
post-hoc analysis can be performed:

    for params in params_store:
        print("likelihood",my_model.likelihood(params))
        print("aic,bic",my_model.aic(params),my_model.bic(params))
    
    np.array(my_model.labels(data,params_store[-1])) ## final predicted labels

The [project repo 'Examples' folder](https://github.com/kasakh/Mixture-Models/tree/master/Mixture_Models) includes more detailed illustrations, as well as a `README.md` for advanced users who want to fit custom mixture models,
or tinker with the settings for the above procedure.

## Background: mixture models

A *mixture model* can be thought of as a probabilistic combination of generative models $f_1$, $f_2$, ..., $f_K$.
If we let $[p_1, \dots, p_K]$ denote the combination weights,
then we generate an observation from the mixture model by:
- using these weights to choose a label $z$ from 1 to K;
- and then generating an observation from $f_z$.

The *mixture model inference problem* is when you have a sample of observations,
and must work backwards to learn the parameters that generated it:
i.e. the probabilities $[p_1, \dots, p_K]$
as well as the parameters for each of the individual models $f_1$, ..., $f_K$.

The primary purpose of this package is to solve this inference problem
by estimating the parameters of a given mixture model.
Models currently implemented include:

- `GMM`: Standard *Gaussian mixture model*
    - `GMM_Constrainted`: GMM with common covariance across components
    - `Mclust`: [MCLUST family](https://sites.stat.washington.edu/raftery/Research/PDF/fraley2003.pdf) of constrained GMMs
- `MFA`: [*Mixture-of-factor analyzers*](https://link.springer.com/article/10.1007/s11222-008-9056-0)
    - `PGMM`: [Parsimonious GMM extension](https://link.springer.com/article/10.1007/s11222-008-9056-0) with constraints
- `TMM`: [*Mixture of t-distributions*](https://www.academia.edu/16834403/Robust_mixture_modelling_using_the_t_distribution?sm=b)

Any *classification problem* can be cast into this framework
by identifying the clusters with mixture model components.
Here is an illustration:

    ### Simulate some dummy data using the built-in function make_pinwheel
    data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                         num_per_class=100, rate=0.4,rs=npr.RandomState(0))
    
    ### Plot the three clusters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:, 0], data[:, 1], 'k.')
    plt.show()

The estimation procedure described above can be run on this data
in order to identify the clusters.

## Note: Automatic differentiation-based inference

Inference for mixture models is usually carried out using Expectation Maximization (EM),
and many software packages have been written to support this.
However, due to the availability of Automatic Differentiation tools,
this package instead uses gradient descent to learn the mixture model parameters
(i.e. step 3 of the above estimation procedure).
There are three main motivations:

1. We can fit models to *high-dimensional data* without any further modeling assumptions.
Fitting models without severe modeling constraints can lead to better clustering performance.
EM based inference cannot scale to high-dimensional data.
2. Gradient descent based inference has *2nd-order optimization routines* like Newton-CG which are faster than 1st-order EM
3. *Unified interface and syntax to fit other classes of mixture models*, no need to jump between R and Python to fit PGMMs/MFAs/etc.

Currently, four main gradient based optimizers are available:

- `"grad_descent"`: Stochastic Gradient Descent (SGD) with momentum
- `"rms_prop"`: Root-mean-squared propagation (RMS-Prop)
- `"adam"`: Adaptive moments (ADAM)
- `"Newton-CG"`: Newton-Conjugate Gradient (Newton CG)

The details about each optimizer and its optional input parameters are given in the PDF in the 'Examples' folder.
The output of `fit` method is the set of all points in the parameter space that the optimizer has traversed during the optimization i.e.  list of parameters with the final entry in the list being the final fitted solution.
We have a detailed notebook `Optimizers_illustration.ipynb` in the 'Examples' folder on Github. 

-------------------------------------------------------------------------------

If you use this package, please consider citing our research as 

 <blockquote>
        <p>@article{kasa2020model,
  title={Model-based Clustering using Automatic Differentiation: Confronting Misspecification and High-Dimensional Data},
  author={Kasa, Siva Rajesh and Rajan, Vaibhav},
  journal={arXiv preprint arXiv:2007.12786},
  year={2020}
}</p>
    </blockquote>


