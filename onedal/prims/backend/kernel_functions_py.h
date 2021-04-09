/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#define NO_IMPORT_ARRAY

#include "data/backend/data.h"
#include "data/backend/utils.h"

#include "oneapi/dal/algo/linear_kernel.hpp"
#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/algo/polynomial_kernel.hpp"

namespace oneapi::dal::python
{
struct linear_kernel_params
{
    double scale;
    double shift;
};

// clas <algorithm>_<act>
class linear_kernel_compute
{
public:
    // from descriptor
    linear_kernel_compute(linear_kernel_params * params);

    // attributes from compute_input
    void compute(PyObject * x, PyObject * y);

    // attributes from compute_result
    PyObject * get_values();

private:
    linear_kernel_params params_;
    linear_kernel::compute_result<> compute_result_;

private:
    static const auto get_descriptor(linear_kernel_params & params, data_type data_type_input);
};

struct rbf_kernel_params
{
    double sigma;
};

// clas <algorithm>_<act>
class rbf_kernel_compute
{
public:
    // from descriptor
    rbf_kernel_compute(rbf_kernel_params * params);

    // attributes from compute_input
    void compute(PyObject * x, PyObject * y);

    // attributes from compute_result
    PyObject * get_values();

private:
    rbf_kernel_params params_;
    rbf_kernel::compute_result<> compute_result_;
};

struct polynomial_kernel_params
{
    double scale;
    double shift;
    double degree;
};

// clas <algorithm>_<act>
class polynomial_kernel_compute
{
public:
    // from descriptor
    polynomial_kernel_compute(polynomial_kernel_params * params);

    // attributes from compute_input
    void compute(PyObject * x, PyObject * y);

    // attributes from compute_result
    PyObject * get_values();

private:
    polynomial_kernel_params params_;
    polynomial_kernel::compute_result<> compute_result_;
};

} // namespace oneapi::dal::python
