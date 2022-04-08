# Installation 

	pip install Mixture-Models


# Mixture Models
This package implements the following mixture models 

1. Gaussian Mixture Models (GMM)

2. [Mixture Factor Models (MFA)](https://link.springer.com/article/10.1007/s11222-008-9056-0)

3. [Parsimonious GMMs (PGMMs)](https://link.springer.com/article/10.1007/s11222-008-9056-0) 

 
In their original papers, the inference for these models is 
carried out using Expectation Maximization (EM). However, due to the availability of 
Automatic Differentiation tools, gradient descent based inference can be carried out 
for these models. These are four main gradient based optimizers available in this version



<ul>
<li>Gradient Descent with Momentum </li>
<li>RMS-Prop</li>
<li>ADAM</li>
<li>Newton-Conjugate Gradient (Newton CG) </li>
</ul>


There are three main motivations for using this package vis-a-vis to existing EM-based packages
<ol>
<li>We can fit GMMs to high-dimensional data without any further 
modeling assumptions. Fitting models without severe modeling constraints can lead 
to better clustering performance. EM based inference cannot scale to high-dimensional data. </li>
<li> Has second-order optimization routines like Newton-CG which is faster than first 
order EM</li>
<li> No Need to jump between R and Python to fit PGMMs or MFAs. All the mixture models are 
 available under one roof and with just one kind of syntax.</li>

</ol>

## Outline of inference procedure
Fitting any of the mixture models mentioned above can be done in 
the following 3 simple steps


    ### Simulate some dummy data using the built-in function make_pinwheel
    data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                         num_per_class=100, rate=0.4,rs=npr.RandomState(0))

We illustrate the 3 steps using GMM model and Newton-CG method. 
    
    ### Choose a model to fit on the data
    GMM_model = GMM(data)

    ### Initialize your model with some parameters    
    init_params = GMM_model.init_params(num_components = 3,scale = 0.5)

    ### Do inference using 
    params_store = GMM_model.fit(init_params,"Newton-CG")

## Details of each step in the inference 
### Choosing the model
Once you have a some input data, choosing a model is relatively straightforward. 
    
    model = GMM(data) ; # For Gaussian Mixture Models (GMMs)
    # model = GMM_conc(data) # For GMMs which share the same covariance matrix for all components
    # model = MFA(data)   # For Mixture of Factor Analyzers
    # model = PGMM(data,constraint="UUU")    # For Parsimonious GMMs, an additional input for the constrained model has to be specified. 

This model initialization step automatically checks if there are any missing/non-finite values in the data. We have detailed complete illustrations
for each of these models in the 'Examples' folder on Github. 

### Initialize the model parameters
    
Each of these models have different input parameters. Further, some parameters have additional constraints. E.g. the mixture weights 
have to be positive and add up to 1. Similarly, the covariance matrix for each of the components has to be positive definite. To enable,
gradient descent for such constrained optimization, we have to **_re-parameterize the inputs_**. More specifically, for the mixture weights, we use 
log-sum-exp trick and for the covariance matrices, we use the decomposition UU^T, where U is a full-rank matrix. 
Details are in section 3 of [this paper](https://arxiv.org/pdf/2007.12786.pdf). 
By reparametrizing the inputs, we are converting the constrained optimization problem into an unconstrained optimization problem. 

For initializing the model with random reparametrized inputs, we use the method `init_params` with the arguments - `num_components` (i.e. number
of clusters to fit) and `scale` (i.e. the size of the mean vectors and covariances matrices). The scale of initialization parameters plays
an important role if the data is not normalized. The default parameter for `scale` is `1.0`

    initial_params = model.init_params(num_components=3,scale=0.5) 



Once the `init_params` method is called, parameters corresponding to the `model` class are randomly initialized and returned as a
dictionary. Here is a sample output for the `initial_params`

    {'log proportions': array([ 0.66579325,  0.35763949, -0.77270015]), 
    'means': array([[-0.00419192,  0.31066799],
           [-0.36004278,  0.13275579],
           [ 0.05427426,  0.00214572]]), 'lower triangles': array([[1., 0.],
           [0., 1.]])}

As you can see, we have `log proportions` (instead of proportions of each component) and `lower triangles` (instead of the covariance matrices).

Once the `initial_params` have been defined, we can manually change it to K-Means initialization or any other user-specified initialization.
Refer the notebook in examples folder for an illustration. 


### Choosing the optimizer

    params_store = model.fit(initial_params,"<optimizers_name>") ## optimizers_name can be Netwon-CG, adam,rms_prop, grad_descent

The details about each optimizer and its optional input parameters are given in the PDF in the 'Examples' folder.  The output of `fit` method is the set of all points in the 
parameter space
that the optimizer has traversed during the optimization i.e.  list of parameters with the final entry in the list being the final 
fitted solution. We have a detailed notebook 'Optimizers_illustration.ipynb' in the 'Examples' folder on Github.  

### Post-hoc Analysis
Once the list of parameters are obtained, we can perform post-hoc analysis as follows:

    for params in params_store:
        print("likelihood",model.likelihood(params))
        print("aic,bic",model.aic(params),model.bic(params))
        
    
    np.array(model.labels(data,params_store[-1])) ## Prints the final labels predicted by the model

If you use this package, please consider citing our research as 

 <blockquote>
        <p>@article{kasa2020model,
  title={Model-based Clustering using Automatic Differentiation: Confronting Misspecification and High-Dimensional Data},
  author={Kasa, Siva Rajesh and Rajan, Vaibhav},
  journal={arXiv preprint arXiv:2007.12786},
  year={2020}
}</p>
    </blockquote>


