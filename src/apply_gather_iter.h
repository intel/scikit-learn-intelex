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

#ifndef _APPLY_GATHER_ITER_INCLUDED_
#define _APPLY_GATHER_ITER_INCLUDED_

#include "cnc4daal.h"
#include "readGraph.h"
#include <cstdlib>
#include <limits>
#include <cmath>

namespace applyGatherIter {

    typedef std::array< int, 2 > pair_t;
    
    template< typename Algo > struct context;

    struct step1 {
        template< typename Context >
        int execute( const pair_t &, Context & ) const;
    };

    struct step2 {
        template< typename Context >
        int execute( const int &, Context & ) const;
    };

    struct applyTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::tag_tuner<>
    {
        template< typename Arg >
        int compute_on( const pair_t & tag, Arg & /*arg*/ ) const
        {
            return std::get<0>(tag) % tuner_base::numProcs();
        }
        template< typename Arg >
        int compute_on( const int & tag, Arg & /*arg*/ ) const
        {
            return tag % tuner_base::numProcs();
        }
        
        int consumed_on( const int tag ) const
        {
            return tag % tuner_base::numProcs();
        }
        
        template< typename T >
        int get_count( const T ) const
        {
            return 1;
        }
    };

    // we do not know the number of iterations
    // we need to keep the data until done
    // -> no get-count (will get GC'ed when tearing down the graph)
    struct i1Tuner : public CnC::hashmap_tuner
    {        
        int consumed_on( const int tag ) const
        {
            return tag % tuner_base::numProcs();
        }
    };
    
    struct i2Tuner : public CnC::hashmap_tuner
    {
        i2Tuner(int nb) : m_n(2*nb) {}
        int m_n;
        
        int consumed_on( const int tag ) const
        {
            return CnC::CONSUMER_ALL;
        }
        
        int get_count( const int ) const
        {
            return m_n;
        }
    };
#ifdef _DIST_
    CNC_BITWISE_SERIALIZABLE(i1Tuner);
    CNC_BITWISE_SERIALIZABLE(i2Tuner);
#endif

    template< int P >
    struct gatherTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::preserve_tuner<int>
    {
        template< typename T, typename Arg >
        int compute_on( const T, Arg & /*arg*/ ) const
        {
            return 0;
        }

        template< typename T >
        int consumed_on( const T ) const
        {
            return P;
        }

        template< typename T >
        int get_count( const T ) const
        {
            return 1;
        }
    };

    struct step1Ctrl {
        step1Ctrl(int nb=0) : m_nBlocks(nb) {}
        std::vector< pair_t > operator()(const int & tag) {
            std::vector< pair_t > v(m_nBlocks);
            for(int i = 0; i<m_nBlocks; ++i) v[i] = {i, tag};
            return v;
        }
    private:
        int m_nBlocks;
    };

    struct mri_tag_manager
    {
        static const size_t N = 1;
        template< typename T >
        T second_input_s1(const T t) {
            t.back() = 0;
            return t;
        }
    };
    
    template< typename Algo >
    class applyGatherContext : public CnC::context< applyGatherContext< Algo > >
    {
    public:
        typedef Algo algo_type;
        typedef step1Ctrl step1_map;
        typedef CnC::identityMap< int > step2_map;
        typedef readGraph< applyTuner, i1Tuner, i2Tuner, Algo::NI > reader_type;
        typedef typename reader_type::in_coll_type read_input_coll_type;
        
        Algo * algo;
        int nBlocks;
        double goal;  // we keep state, this is not pure!
        double accuracyThreshold;

        read_input_coll_type readInput;
        reader_type * reader;
        i2Tuner i2tuner;

        CnC::dc_step_collection< int, pair_t, step1_map, step1, applyTuner > step_1;
        CnC::dc_step_collection< int, int, step2_map, step2, gatherTuner<0> > step_2;

        CnC::item_collection< int, typename Algo::iomstep1Local_type::input1_type, i1Tuner > step1Input1;
        CnC::item_collection< int, data_management::NumericTablePtr, i2Tuner > step1Input2;
        CnC::item_collection< pair_t, typename Algo::iomstep2Master_type::input1_type, gatherTuner<0> > step2Input;
        CnC::item_collection< int, typename Algo::iomstep2Master_type::result_type, gatherTuner<CnC::CONSUMER_ALL> > result;

        applyGatherContext(Algo * a = NULL, int numBlocks = 0)
            : algo(a),
              nBlocks(numBlocks),
              readInput(*this, "files"),
              reader(NULL),
              goal(std::numeric_limits<double>::max()),
              accuracyThreshold(0.0),
              i2tuner(numBlocks),
              step_1( *this, "apply" ),
              step_2( *this, "gather" ),
              step1Input1(*this, "step1Input1"),
              step1Input2(*this, "step1Input2", i2tuner),
              step2Input( *this, "step2Input"),
              result( *this, "result")
        {
            step_1.consumes( step1Input1 );
            step_1.consumes( step1Input2, step1_map(nBlocks) );
            step_1.produces( step2Input );
            step_2.consumes( step2Input );
            step_2.consumes( step1Input2, step2_map() );
            step_2.produces( step1Input2 );
            step_2.produces( result );
            reader = new reader_type(*this, readInput, step1Input1, step1Input2);
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this);
            if(algo) {
                accuracyThreshold = use_default(algo->_accuracyThreshold)
                    ? typename Algo::algob_type::ParameterType(algo->_nClusters, algo->_maxIterations).accuracyThreshold
                    : algo->_accuracyThreshold;
            }
        }
        ~applyGatherContext()
        {
            delete reader;
        }
#ifdef _DIST_
        void serialize(CnC::serializer & ser)
        {
            if(ser.is_unpacking()) {
                assert(algo == NULL);
                algo = new Algo;
            }
            ser & (*algo) & nBlocks & goal & accuracyThreshold & i2tuner;
            if(ser.is_cleaning_up()) delete algo;
        }
#endif

    };

#if 0
    void CHECK(const daal::data_management::NumericTablePtr & t)
    {
        if(!t) return;
        auto ni = t->getNumberOfRows();
        auto nj = t->getNumberOfColumns();
        for(auto i=0; i<ni; ++i)
            for(auto j=0; j<nj; ++j)
                assert(std::isfinite(t->daal::data_management::NumericTable::getValue<double>(j,i)));
    }
#else
# define CHECK(_x)
#endif
    
    template< typename Context >
    int step1::execute(const pair_t & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input1_type pData1;
        ctxt.step1Input1.get(std::get<0>(tag), pData1);
        CHECK(pData1);
        typename Context::algo_type::iomstep1Local_type::input2_type pData2;
        ctxt.step1Input2.get(std::get<1>(tag), pData2);
        CHECK(pData2);

        typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData1, pData2);
        CHECK(res->get(algorithms::kmeans::partialSums));
        CHECK(res->get(algorithms::kmeans::nObservations));
        CHECK(res->get(algorithms::kmeans::partialObjectiveFunction));
        CHECK(res->get(algorithms::kmeans::partialAssignments));

        ctxt.step2Input.put(tag, res);
        return 0;
    }

    template< typename Context >
    int step2::execute(const int & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input2_type pData2;
        ctxt.step1Input2.get(tag, pData2);
        CHECK(pData2);
        std::vector< typename Context::algo_type::iomstep2Master_type::input1_type > inp(ctxt.nBlocks);
        for( int i = 0; i < ctxt.nBlocks ; i++ ) {
            ctxt.step2Input.get({i, tag}, inp[i]);
            CHECK(inp[i]->get(algorithms::kmeans::partialSums));
            CHECK(inp[i]->get(algorithms::kmeans::nObservations));
            CHECK(inp[i]->get(algorithms::kmeans::partialObjectiveFunction));
            CHECK(inp[i]->get(algorithms::kmeans::partialAssignments));
        }

        typename Context::algo_type::iomstep2Master_type::result_type res = ctxt.algo->run_step2Master(inp);

        if(tag < ctxt.algo->_maxIterations) {
            double new_goal = res->get(daal::algorithms::kmeans::goalFunction)->daal::data_management::NumericTable::getValue<double>(0, 0);
            if(std::abs(ctxt.goal - new_goal) > ctxt.accuracyThreshold) {
                CHECK(res->get(daal::algorithms::kmeans::centroids));
                ctxt.step1Input2.put(tag+1, res->get(daal::algorithms::kmeans::centroids));
                ctxt.goal = new_goal;
                return 0;
            }
        }
        daal::data_management::NumericTablePtr maxittab(new daal::data_management::HomogenNumericTable<int>(1, 1, daal::data_management::NumericTable::doAllocate, (int)tag));
        res->set(daal::algorithms::kmeans::nIterations, maxittab);
        ctxt.result.put(0, res);
        return 0;
    }

    template< typename Algo >
    struct applyGatherIter
    {
        typedef Algo manager_type;
        typedef applyGatherContext< Algo > context_type;

        static typename Algo::iomstep2Master_type::result_type
        compute(Algo & algo, const TableOrFList & input, const TableOrFList & input2)
        {
            algo._assignFlag = "false";
            int pid = CnC::tuner_base::myPid();
            int stride = CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1;
            int nblocks = stride;
            if( ! (input.table || input.file.size()) ) {
                nblocks *= input.flist.size() ? input.flist.size() : input.tlist.size();
            }

            context_type ctxt(&algo, nblocks);
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(pid == 0) {
                assert(input2.table);
                CHECK(input2.table);
                ctxt.step1Input2.put(0, input2.table);
            }
            if(input.table) {
                CHECK(input.table);
                ctxt.step1Input1.put(pid, input.table);
            } else if(input.file.size()) {
                ctxt.readInput.put(pid, {input.file});
            } else {
                assert((input.flist.size() == 0) != (input.tlist.size() == 0));
                for(int i = 0; i < input.flist.size(); ++i) {
                    ctxt.readInput.put(pid + i * stride, typename context_type::reader_type::input_type({input.flist[i]}));
                }
                for(int i = 0; i < input.tlist.size(); ++i) {
                    CHECK(input.tlist[i]);
                    ctxt.step1Input1.put(pid + i * stride, input.tlist[i]);
                }
            }
            ctxt.wait();

            typename Algo::iomstep2Master_type::result_type res;
            if(pid == 0 || CnC::Internal::distributor::distributed_env()) {
                ctxt.result.get(0, res);
            }

            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }
    };

} // namespace applyGatherIter


#endif // _APPLY_GATHER_ITER_INCLUDED_
