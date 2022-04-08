from setuptools import setup

setup(
   name='Mixture_Models',
   version='0.1',
   description='For fitting mixture models with Gradient Descent based optimizers',
   author='Kasa',
   author_email='kasa@u.nus.edu',
   packages=['Mixture_Models'],  #same as name
   install_requires=['scipy','sklearn', 'matplotlib', 'autograd'], #external packages as dependencies
)
