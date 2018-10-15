import os
import sys
test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(os.path.dirname(test_path), "examples")
sys.path.insert(0, examples_path)
os.chdir(examples_path)

import unittest
import numpy as np
from daal4py import __daal_link_version__ as dv
daal_version = tuple(map(int, (dv[0:4], dv[4:8])))
import sklearn.metrics

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=np.float64).values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


class Test(unittest.TestCase):
    def test_adagrad_mse_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "adagrad_mse_batch.csv"), range(1))
        from adagrad_mse_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_correlation_distance_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "correlation_distance_batch.csv"), range(1))
        from correlation_distance_batch import main as get_results
        result = get_results()
        r = result.correlationDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    def test_cosine_distance_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "cosine_distance_batch.csv"), range(1))
        from cosine_distance_batch import main as get_results
        result = get_results()
        r = result.cosineDistance
        self.assertTrue(np.allclose(np.array([[np.amin(r)],[np.amax(r)],[np.mean(r)],[np.average(r)]]), testdata))

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_classification_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "decision_forest_classification_batch.csv"), range(1))
        from decision_forest_classification_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    @unittest.skipIf(daal_version < (2019, 1), "not supported in this library version")
    def test_decision_forest_regression_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "decision_forest_regression_batch.csv"), range(1))
        from decision_forest_regression_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_classification_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "decision_tree_classification_batch.csv"), range(1))
        from decision_tree_classification_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_decision_tree_regression_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "decision_tree_regression_batch.csv"), range(1))
        from decision_tree_regression_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_classification_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "gradient_boosted_classification_batch.csv"), range(1))
        from gradient_boosted_classification_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_gradient_boosted_regression_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "gradient_boosted_regression_batch.csv"), range(1))
        from gradient_boosted_regression_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(sklearn.metrics.mean_squared_error(predict_result.prediction, testdata) < 1e-2)

    def test_kmeans_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "kmeans_batch.csv"), range(20))
        from kmeans_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.centroids, testdata))

    def test_lbfgs_cr_entr_loss_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "lbfgs_cr_entr_loss_batch.csv"), range(1))
        from lbfgs_cr_entr_loss_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_lbfgs_mse_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "lbfgs_mse_batch.csv"), range(1))
        from lbfgs_mse_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_linear_regression_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "linear_regression_batch.csv"), range(2))
        from linear_regression_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_binary_dense_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "log_reg_binary_dense_batch.csv"), range(1))
        from log_reg_binary_dense_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_log_reg_dense_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "log_reg_dense_batch.csv"), range(1))
        from log_reg_dense_batch import main as get_results
        (_, predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_low_order_moms_dense_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "low_order_moms_dense_batch.csv"), range(10))
        from low_order_moms_dense_batch import main as get_results
        res = get_results()
        r = np.vstack((
            res.minimum,
            res.maximum,
            res.sum,
            res.sumSquares,
            res.sumSquaresCentered,
            res.mean,
            res.secondOrderRawMoment,
            res.variance,
            res.standardDeviation,
            res.variation
        ))
        self.assertTrue(np.allclose(r, testdata))

    def test_multivariate_outlier_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "multivariate_outlier_batch.csv"), range(1))
        from multivariate_outlier_batch import main as get_results
        ( _,result) = get_results()
        self.assertTrue(np.allclose(result.weights, testdata))

    def test_naive_bayes_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "naive_bayes_batch.csv"), range(1))
        from naive_bayes_batch import main as get_results
        (predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_pca_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "pca_batch.csv"), range(10))
        from pca_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.eigenvectors, testdata))

    def test_ridge_regression_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "ridge_regression_batch.csv"), range(2))
        from ridge_regression_batch import main as get_results
        (predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_sgd_logistic_loss_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "sgd_logistic_loss_batch.csv"), range(1))
        from sgd_logistic_loss_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_sgd_mse_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "sgd_mse_batch.csv"), range(1))
        from sgd_mse_batch import main as get_results
        result = get_results()
        self.assertTrue(np.allclose(result.minimum, testdata))

    def test_svd_batch(self):
        from svd_batch import main as get_results
        (data, result) = get_results()
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))

    def test_svm_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "svm_batch.csv"), range(1))
        from svm_batch import main as get_results
        (predict_result, _) = get_results()
        self.assertTrue(np.allclose(np.sign(predict_result.prediction), testdata))

    def test_svm_multiclass_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "svm_multiclass_batch.csv"), range(1))
        from svm_multiclass_batch import main as get_results
        (predict_result, _) = get_results()
        self.assertTrue(np.allclose(predict_result.prediction, testdata))

    def test_univariate_outlier_batch(self):
        testdata = read_csv(os.path.join(unittest_data_path, "univariate_outlier_batch.csv"), range(1))
        from univariate_outlier_batch import main as get_results
        ( _,result) = get_results()
        self.assertTrue(np.allclose(result.weights, testdata))


if __name__ == '__main__':
    unittest.main()
