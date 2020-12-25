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
import ray
import modin.pandas as pd


@ray.remote(num_cpus=1)
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


def distributed_k_means_fit(X, n_clusters, max_iter):
    num_nodes = len(ray.nodes())

    actors = [
        KMeansRowPartitionsActor.options(num_cpus=1, resources={node: 1.0}).remote(node)
        for node in ray.cluster_resources()
        if "node" in node
    ]

    assert num_nodes == len(
        actors
    ), f"number of nodes {num_nodes} is not equal to number of actors {len(actors)}"

    row_partitions = unwrap_row_partitions(X)

    row_parts_last_idx = (
        len(row_partitions) // num_nodes
        if len(row_partitions) % num_nodes == 0
        else len(row_partitions) // num_nodes + 1
    )

    i = 0
    for actor in actors:
        actor.set_row_parts._remote(
            args=(
            row_partitions[
                slice(i, i + row_parts_last_idx)
                if i + row_parts_last_idx < len(row_partitions)
                else slice(i, len(row_partitions))
            ])
        )
        i += row_parts_last_idx

    context = RayContext()
    pyccl = PyOneCCL(context.get_world_size(), context.current_node_id())

    # KMeans random initialization
    centroids = X.sample(n=n_clusters).to_numpy()

    result = np.array([]).reshape(3, 0)

    for actor in actors:
        tmp = actor._kmeans_dal_compute_with_init_centers._remote(
            args=(centroids, n_clusters, max_iter, pyccl),
            num_returns=3,
        )
        result = np.concatenate((result, np.array([tmp]).T), axis=1)

    # retrieving data via object refs
    centroids, inertia, assignments = (
        ray.get(list(result[0]))[0],
        ray.get(list(result[1]))[0],
        pd.DataFrame(ray.get(list(result[2]))[0]),
    )

    return centroids, assignments, inertia, max_iter
