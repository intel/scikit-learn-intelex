- Without editing the code of a scikit-learn application by using the following command line flag::

    python -m sklearnex my_application.py

- Directly from the script::

    from sklearnex import patch_sklearn
    patch_sklearn()

- Through :ref:`global patching <global_patching>` to enable patching for your scikit-learn installation for all further runs::

    python sklearnex.glob patch_sklearn