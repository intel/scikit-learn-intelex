# :snake: Intel(R) Extension for Scikit-learn* notebooks

This folder contains examples of python notebooks that use Intel(R) extension for Scikit-learn for popular datasets.  

#### :rocket: Jupyter startup guide
You can use python notebooks with the help of Jupyter* notebook to run the following files:

```bash
conda install -c conda-forge notebook scikit-learn-intelex
```  
or  
```bash
pip install notebook scikit-learn-intelex
```  
Run Jupyter after installation:
```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```  

#### :pencil: Table of contents

| Algorithm               | Workload       | Notebook       | Scikit-learn estimator|
| :----------------------:| :------------: | :-------------:| :-------------------:|
|    LogisticRegression  |    CIFAR-100    | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/logistictic_regression_cifar.ipynb)    | [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
|          SVC           |     Adult       | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/svc_adult.ipynb) | [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| KNeighborsClassifier   |       MNIST     | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/knn_mnist.ipynb) |    [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
|        NuSVR           | Medical charges | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/nusvr_medical_charges.ipynb) | [sklearn.svm.NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html) |
| RandomForestRegressor  |     Yolanda     | [View source on GitHub](https://github.com/intel/scikit-learn-intelex/blob/master/examples/notebooks/random_forest_yolanda.ipynb) | [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) |
