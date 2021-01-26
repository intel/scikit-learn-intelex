#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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
#******************************************************************************/

import daal4py
from daal4py.engines.row_partition_actor import RowPartitionsActor
from daal4py.engines.ray.ray_context import RayContext
from daal4py.sklearn.utils.pyoneccl import PyOneCCL

from modin.developer import unwrap_row_partitions

import numpy as np
from distributed.client import get_client
import modin.pandas as pd


class KMeansRowPartitionsActor(RowPartitionsActor):
    def __init__(self, node):
        super().__init__(node)

    def _kmeans_dal_compute_with_init_centers(
        self, centroids, n_clusters, n_iters, pyccl
    ):
        pyccl.set_env()

        data_np = np.ascontiguousarray(self.get_row_parts().to_numpy())
        kmeans = daal4py.kmeans(n_clusters, n_iters, distributed=True)
        result = kmeans.compute(data_np, centroids)

        kmeans = daal4py.kmeans(n_clusters, 0, assignFlag=True)
        assignments = kmeans.compute(data_np, result.centroids).assignments

        result = (result.centroids, result.objectiveFunction, assignments) if daal4py.my_procid()==0 else (np.asarray([]), 0, assignments)

        return result


def dask_distributed_k_means_fit(X, n_clusters, max_iter):

    actors = []
    dask_client = get_client()
    dask_workers = dask_client.scheduler_info()["workers"]

    ips = list(set([dask_workers[worker]["host"] for worker in dask_workers.keys()]))
    for ip in ips:
        actors.append(dask_client.submit(RowPartitionsActor, None, workers=set([ip]), actor=True))
    actors = [actor.result() for actor in actors]
   
    row_partitions = unwrap_row_partitions(X)

    num_nodes = len(actors)

    row_parts_last_idx = (
        len(row_partitions) // num_nodes
        if len(row_partitions) % num_nodes == 0
        else len(row_partitions) // num_nodes + 1
    )

    i = 0
    for actor in actors:
        actor.set_row_parts(
            [r.result() for r in row_partitions[
                slice(i, i + row_parts_last_idx)
                if i + row_parts_last_idx < len(row_partitions)
                else slice(i, len(row_partitions))
            ]]
        )
        i += row_parts_last_idx

    context = DaskContext()
    pyccl = PyOneCCL(context.get_world_size(), context.current_node_id())

    # KMeans random initialization
    centroids = X.sample(n=n_clusters).to_numpy()

    result = np.array([]).reshape(3, 0)

    for actor in actors:
        tmp = actor._kmeans_dal_compute_with_init_centers(centroids, n_clusters, max_iter, ccl_context, context)
        futures = [dask_client.submit(lambda l: l[i], tmp.result()) for i in range(3)]
        result = np.concatenate((result, np.array([futures]).T), axis=1)

    # retrieving data via object refs
    centroids, inertia, assignments = (
        result[0][0].result(),
        result[1][0].result(),
        pd.DataFrame(result[2][0].result()),
    )

    return centroids, assignments, inertia, max_iter
