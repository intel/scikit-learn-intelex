::

    from sklearnex import patch_sklearn
    # The names match scikit-learn estimators
    patch_sklearn(["SVC", "DBSCAN"])