python -c "import daal4py"
python -m unittest discover -v -s tests -p test*.py
pytest --pyargs daal4py/sklearn/
python examples/run_examples.py
python -m daal4py examples/sycl/sklearn_sycl.py
