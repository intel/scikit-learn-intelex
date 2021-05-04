#===============================================================================
# Copyright 2014-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import daal4py as d4p
import os

if d4p.__has_dist__:
    import unittest
    import numpy as np
    from test_examples import np_read_csv, add_test

    class Base():
        def test_svd_spmd(self):
            import svd_spmd as ex
            (data, result) = self.call(ex)
            self.assertTrue(
                np.allclose(
                    data,
                    np.matmul(
                        np.matmul(
                            result.leftSingularMatrix,
                            np.diag(result.singularValues[0])
                        ),
                        result.rightSingularMatrix
                    )
                )
            )

        def test_qr_spmd(self):
            import qr_spmd as ex
            (data, result) = self.call(ex)
            self.assertTrue(np.allclose(data, np.matmul(result.matrixQ, result.matrixR)))

        def test_kmeans_spmd(self):
            nClusters = 10
            maxIter = 25

            data = np.loadtxt("./data/distributed/kmeans_dense.csv", delimiter=',')

            rpp = int(data.shape[0] / d4p.num_procs())
            spmd_data = data[rpp * d4p.my_procid():rpp * d4p.my_procid() + rpp, :]

            for init_method in ['plusPlusDense',
                                'parallelPlusDense',
                                'deterministicDense']:
                batch_init_res = d4p.kmeans_init(nClusters=nClusters,
                                                 method=init_method).compute(data)
                spmd_init_res = d4p.kmeans_init(nClusters=nClusters,
                                                method=init_method,
                                                distributed=True).compute(spmd_data)

                if init_method in ['parallelPlusDense']:
                    print("Warning: It is well known "
                          "that results of parallelPlusDense init "
                          "does not match with batch algorithm")
                else:
                    reason = "Initial centroids with " + init_method
                    reason += " does not match with batch algorithm"
                    self.assertTrue(
                        np.allclose(batch_init_res.centroids, spmd_init_res.centroids),
                        reason
                    )

                batch_res = d4p.kmeans(
                    nClusters=nClusters,
                    maxIterations=maxIter).compute(data, batch_init_res.centroids)
                spmd_res = d4p.kmeans(
                    nClusters=nClusters,
                    maxIterations=maxIter,
                    distributed=True).compute(spmd_data, spmd_init_res.centroids)

                if init_method in ['parallelPlusDense']:
                    print("Warning: It is well known "
                          "that results of parallelPlusDense init "
                          "does not match with batch algorithm")
                else:
                    reason = "Final centroids with " + init_method
                    reason += " does not match with batch algorithm"
                    self.assertTrue(
                        np.allclose(batch_res.centroids, spmd_res.centroids),
                        reason
                    )

        def test_dbscan_spmd(self):
            epsilon = 0.04
            minObservations = 45
            data = np_read_csv(os.path.join(".", 'data', 'batch', 'dbscan_dense.csv'))

            batch_algo = d4p.dbscan(minObservations=minObservations,
                                    epsilon=epsilon,
                                    resultsToCompute='computeCoreIndices')
            batch_result = batch_algo.compute(data)

            rpp = int(data.shape[0] / d4p.num_procs())
            node_stride = rpp * d4p.my_procid()
            node_range = range(node_stride, node_stride + rpp)
            node_data = data[node_range, :]

            spmd_algo = d4p.dbscan(minObservations=minObservations,
                                   epsilon=epsilon, distributed=True)
            spmd_result = spmd_algo.compute(node_data)

            # clusters can get different indexes in batch and spmd algos,
            # to compare them we should take care about it
            cluster_index_dict = {}
            for i in node_range:
                # border points assignments can be different
                # with different amount of nodes but cores are the same
                if i in batch_result.coreIndices:
                    right = spmd_result.assignments[i - node_stride][0]
                    if not batch_result.assignments[i][0] in cluster_index_dict:
                        cluster_index_dict[batch_result.assignments[i][0]] = right
                    left = cluster_index_dict[batch_result.assignments[i][0]]
                    self.assertTrue(
                        left == right
                    )

    gen_examples = [
        ('covariance_spmd', 'covariance.csv', 'covariance'),
        ('low_order_moms_spmd', 'low_order_moms_dense_batch.csv',
         lambda r: np.vstack((r.minimum, r.maximum, r.sum, r.sumSquares,
                              r.sumSquaresCentered, r.mean, r.secondOrderRawMoment,
                              r.variance, r.standardDeviation, r.variation))),
    ]

    for example in gen_examples:
        add_test(Base, *example)

    class Test(Base, unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            d4p.daalinit()

        @classmethod
        def tearDownClass(cls):
            d4p.daalfini()

        def call(self, ex):
            return ex.main()

    if __name__ == '__main__':
        unittest.main()
