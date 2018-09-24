/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
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

#ifndef _HLAPI_DISTR_H_INCLUDED_
#define _HLAPI_DISTR_H_INCLUDED_

#ifdef _WIN32
#define NOMINMAX
#endif
#include "daal4py.h"

#ifdef _DIST_
#include <tuple>
#include "cnc4daal.h"

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// serialization of a TableOrFlist
inline void serialize(CnC::serializer & ser, TableOrFList & obj)
{
    ser & obj.table & obj.flist;
}

// Input/Output manager for simple algos with 2 inputs
// abstracts from input/output types
// also defines how to get results and finalize
template< typename A, typename I1, typename I2, typename O >
struct IOManager2 : public IOManager< A, I1, O >
{
    typedef I2 input2_type;
    typedef std::tuple< I1, input2_type > input_type;
};

// Input/Output manager for simple algos with 3 inputs
// abstracts from input/output types
// also defines how to get results and finalize
template< typename A, typename I1, typename I2, typename I3, typename O >
struct IOManager3 : public IOManager2< A, I1, I2, O >
{
    typedef I3 input3_type;
    typedef std::tuple< I1, I2, input3_type > input_type;
};

// Input/Output manager for simple algos with 1 input and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I, typename O, typename E, int P>
struct IOManagerSingle : public IOManager< A, I, O >
{
    static O getResult(A & algo)
    {
        return algo.getResult()->get(static_cast< E >(P));
    }
};

// Input/Output manager for simple algos with 2 inputs and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename O, typename E, int P>
struct IOManager2Single : public IOManagerSingle< A, I1, O, E, P >
{
    typedef I2 input2_type;
    typedef std::tuple< I1, I2 > input_type;
};

// Input/Output manager for simple algos with 3 inputs and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename I3, typename O, typename E, int P>
struct IOManager3Single : public IOManager2Single< A, I1, I2, O, E, P >
{
    typedef I3 input3_type;
    typedef std::tuple< I1, I2, I3 > input_type;
};


// Input/Output manager for intermediate steps with 1 input
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I, typename O>
struct PartialIOManager
{
    typedef O result_type;
    typedef I input1_type;
    typedef std::tuple< input1_type > input_type;

    static result_type getResult(A & algo)
    {
        return daal::services::staticPointerCast<typename result_type::ElementType>(algo.getPartialResult());
    }
    static bool needsFini()
    {
        return false;
    }
};

// Input/Output manager for intermediate steps with 2 inputs
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename O>
struct PartialIOManager2 : public PartialIOManager< A, I1, O >
{
    typedef I2 input2_type;
    typedef std::tuple< I1, I2 > input_type;
};

// Input/Output manager for intermediate steps with 3 inputs
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename I3, typename O>
struct PartialIOManager3 : public PartialIOManager2< A, I1, I2, O >
{
    typedef I3 input3_type;
    typedef std::tuple< I1, I2, I3 > input_type;
};

// Input/Output manager for intermediate steps with 1 input and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I, typename O, typename E, int P>
struct PartialIOManagerSingle : public PartialIOManager< A, I, O >
{
    static typename PartialIOManager< A, I, O >::result_type getResult(A & algo)
    {
        return algo.getPartialResult()->get(static_cast< E >(P));
    }
};

// Input/Output manager for intermediate steps with 2 inputs and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename O, typename E, int P>
struct PartialIOManager2Single : public PartialIOManagerSingle< A, I1, O, E, P >
{
    typedef I2 input2_type;
    typedef std::tuple< I1, I2 > input_type;
};

// Input/Output manager for intermediate steps with 3 inputs and a single fixed result-component
// abstracts from input/output types
// also defines how to get results and finalize
template<typename A, typename I1, typename I2, typename I3, typename O, typename E, int P>
struct PartialIOManager3Single : public PartialIOManager2Single< A, I1, I2, O, E, P >
{
    typedef I3 input3_type;
    typedef std::tuple< I1, I2, I3 > input_type;
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

#include "map_reduce.h"
#include "map_reduce_iter.h"
#include "apply_gather.h"
#include "dkmi.h"

#endif // _DIST_
#endif // _HLAPI_DISTR_H_INCLUDED_
