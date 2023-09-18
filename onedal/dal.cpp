/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "onedal/common.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

/* common */
ONEDAL_PY_INIT_MODULE(policy);
#ifdef ONEDAL_DATA_PARALLEL_SPMD
ONEDAL_PY_INIT_MODULE(spmd_policy);
#endif

/* datatypes*/
ONEDAL_PY_INIT_MODULE(table);
ONEDAL_PY_INIT_MODULE(table_metadata);

/* primitives */
ONEDAL_PY_INIT_MODULE(get_tree);
ONEDAL_PY_INIT_MODULE(covariance);
ONEDAL_PY_INIT_MODULE(linear_kernel);
ONEDAL_PY_INIT_MODULE(rbf_kernel);
ONEDAL_PY_INIT_MODULE(polynomial_kernel);
ONEDAL_PY_INIT_MODULE(sigmoid_kernel);

/* algorithms */
ONEDAL_PY_INIT_MODULE(dbscan);
ONEDAL_PY_INIT_MODULE(ensemble);
ONEDAL_PY_INIT_MODULE(decomposition);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
ONEDAL_PY_INIT_MODULE(basic_statistics);
ONEDAL_PY_INIT_MODULE(linear_model);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans_init);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans);
ONEDAL_PY_INIT_MODULE(kmeans_common);
ONEDAL_PY_INIT_MODULE(neighbors);
ONEDAL_PY_INIT_MODULE(svm);

#ifdef ONEDAL_DATA_PARALLEL
PYBIND11_MODULE(_onedal_py_dpc, m) {
#ifdef ONEDAL_DATA_PARALLEL_SPMD
    init_spmd_policy(m);
#endif
#else
PYBIND11_MODULE(_onedal_py_host, m) {
#endif
    init_policy(m);
    init_table(m);
    init_table_metadata(m);

    init_covariance(m);
    init_linear_kernel(m);
    init_rbf_kernel(m);
    init_polynomial_kernel(m);
    init_sigmoid_kernel(m);
    init_get_tree(m);

    init_dbscan(m);
    init_decomposition(m);
    init_ensemble(m);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
    init_basic_statistics(m);
    init_linear_model(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans_init(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans(m);
    init_kmeans_common(m);
    init_neighbors(m);
    init_svm(m);
}

} // namespace oneapi::dal::python
