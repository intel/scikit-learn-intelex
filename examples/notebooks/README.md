# :snake: Intel(R) Extension for Scikit-learn* notebooks

This folder contains examples of Jupyter* notebooks that use Intel(R) extension for Scikit-learn for popular datasets.  
You need Jupyter* notebook to run the following .ipynb files:

```bash
conda install -c conda-forge notebook scikit-learn-intelex
```  
or  
```bash
pip install notebook scikit-learn-intelex
```  

#### :pencil: Table of contents

| Algorithm               | Workload       | Notebook       | Sckit-learn estimator|
| :----------------------:| :------------: | :-------------:| :-------------------:|
|    LogisticRegression  |    CIFAR-100    | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/logistictic_regression_cifar.ipynb)    | [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
|          SVC           |     Adult       | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/svc_adult.ipynb) | [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| KNeighborsClassifier   |       MNIST     | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/knn_mnist.ipynb) |    [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
|        NuSVR           | Medical charges | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/nusvr_medical_charges.ipynb) | [sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html) |
| RandomForestRegressor  |     Yolanda     | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/random_forest_yolanda.ipynb) | [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |

Training and inference times are measured using the [**%%time**](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time) function. This shows 2 times:  
- CPU times: **sys** - the operating system CPU time due to system calls from the process;  
- CPU times: **user** - the process CPU time (contains all cores);  
- **Wall time** - time of cell computing
