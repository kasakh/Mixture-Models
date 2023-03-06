## Utility functions reused across various tests

from Mixture_Models import checkers
import autograd.numpy as np
import json
import os

def check_array_equal(actual,expected,exact=False):
    checkers.check_dim(actual,np.shape(expected))
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

def init_MM(MMclass,data,seed,init_params_args,expected,**kwargs):
    MM = MMclass(data,**kwargs)
    np.random.seed(seed)
    init_params = MM.init_params(**init_params_args)
    check_arraydict_equal(init_params,expected)
    return MM, init_params

def check_fit(MM,init_params,opt_routine,fit_args,expected_dim,expected_metrics):
    params_store = MM.fit(init_params,opt_routine,**fit_args)
    checkers.check_dim(params_store,(expected_dim,))
    for metric in expected_metrics:
        check_metric_equal(MM,metric,params_store,expected_metrics[metric])
    return params_store

def check_fromfile(MM,init_params,opt_routine,fit_args,filepath,metrics):
    params_store = MM.fit(init_params,opt_routine,**fit_args)
    actual_metrics = {metric:[getattr(MM,metric)(param) for param in params_store] for metric in metrics}
    actual_labels = np.array(MM.labels(MM.data,params_store[-1]))
    #create_run([params_store,actual_metrics, actual_labels],filepath)      #uncomment this line when testing for the first time
    with open(os.path.join('tests',filepath)) as param_file:
        expected_params_store, expected_metrics, expected_labels = json.load(param_file)
    assert len(params_store) == len(expected_params_store)
    for i in range(len(params_store)):
        check_arraydict_equal(params_store[i],expected_params_store[i])
    check_arraydict_equal(actual_metrics,expected_metrics)
    check_array_equal(actual_labels,expected_labels,exact=True)
    return params_store

def create_run(run,filepath):
    # Numpy serializer code from https://stackoverflow.com/a/47626762
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    with open(os.path.join('tests',filepath),'w') as param_file:
        json.dump(run,param_file,cls=NumpyEncoder)