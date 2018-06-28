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

#ifndef _MAP_RED_ITER_INCLUDED_
#define _MAP_RED_ITER_INCLUDED_

#include "cnc4daal.h"
#include "readGraph.h"
#include "map_reduce_graph.h"
#include <cstdlib>
#include <limits>
#include <cmath>

namespace mapReduceIter {

    template< typename Algo > struct context;

    typedef std::array< int, 2 > s1i_tag_type;
    
    // we do not know the number of iterations
    // we need to keep the data until done
    // -> no get-count (will get GC'ed when tearing down the graph)
    struct i1Tuner : public mapReduce::mapRedTuner< 1 >
    {
        using mapReduce::mapRedTuner< 1 >::mapRedTuner;

        template< typename T >
        int get_count(const T &) const
        {
            return CnC::NO_GETCOUNT;
        }
    };
    
    
    struct i2Tuner : public CnC::hashmap_tuner
    {
        i2Tuner(int nb) : m_n(nb) {}
        int m_n;

        int consumed_on(const s1i_tag_type &) const
        {
            return CnC::CONSUMER_ALL;
        }
        
        int get_count(const s1i_tag_type &) const
        {
            return m_n;
        }
    };

    struct step1Ctrl {
        step1Ctrl(int nb=0) : m_nBlocks(nb) {}
        template< typename T >
        std::vector< s1i_tag_type > operator()(const T & tag) {
            std::vector< s1i_tag_type > v(m_nBlocks);
            for(int i = 0; i<m_nBlocks; ++i) v[i] = {i, std::get<1>(tag)};
            return v;
        }
    private:
        int m_nBlocks;
    };

    struct mri_tag_manager
    {
        mri_tag_manager(int nb=-1)
            : s1i1_tuner(nb),
              s1i2_tuner(nb),
              tuner(nb),
              s1i2_map(nb),
              s1i1_map()
        {}

        static const size_t N = 1;
        
        typedef mapReduce::tagger<N> tagger_type;
        typedef i1Tuner s1i1_tuner_type;
        typedef i2Tuner s1i2_tuner_type;
        typedef mapReduce::mapRedTuner<N> tuner_type;
        s1i1_tuner_type s1i1_tuner;
        s1i2_tuner_type s1i2_tuner;
        tuner_type tuner;
        
        typedef step1Ctrl step1_map_type;
        step1_map_type s1i2_map;
        CnC::no_mapper s1i1_map;

        template< typename T >
        static inline T input_s1(const int n, const T & t) {
            if(n == 0)      return {std::get<0>(t), 0};
            else if(n == 1) return {0, std::get<1>(t)};
            return {-1, -1};
        }
    };
    
#ifdef _DIST_
    CNC_BITWISE_SERIALIZABLE(step1Ctrl);
    CNC_BITWISE_SERIALIZABLE(i1Tuner);
    CNC_BITWISE_SERIALIZABLE(i2Tuner);
    CNC_BITWISE_SERIALIZABLE(mri_tag_manager);
#endif
    
    template< typename Algo >
    class mapRedIterContext : public CnC::context< mapRedIterContext< Algo > >
    {
    public:
        struct Fini {
            int execute(const int, mapRedIterContext< Algo > &) const;
        };
        
        typedef Algo algo_type;

        typedef mri_tag_manager tag_manager_type;
        typedef tag_manager_type::tagger_type::step1_tag_type step1_tag_type;
        typedef readGraph< step1_tag_type, mapReduce::mapRedTuner<1>, i1Tuner, i2Tuner, Algo::NI > reader_type;
        typedef typename reader_type::in_coll_type read_input_coll_type;

        typedef mapReduce::mapReduceGraph< mapRedIterContext< Algo >, tag_manager_type > mr_graph_type;
        typedef typename mr_graph_type::step1_input1_coll_type step1_input1_coll_type;
        typedef typename mr_graph_type::step1_input2_coll_type step1_input2_coll_type;
        typedef typename mr_graph_type::step2_input_coll_type step2_input_coll_type;
        typedef typename mr_graph_type::result_coll_type result_coll_type;
        
        Algo * algo;
        int nBlocks;
        double goal;  // we keep state, this is not pure!
        double accuracyThreshold;
        mri_tag_manager tm;
        
        read_input_coll_type readInput;
        step1_input1_coll_type step1Input1;
        step1_input2_coll_type step1Input2;
        result_coll_type result;
        
        mr_graph_type * mr_graph;
        reader_type * reader;

        mapRedIterContext(Algo * a = NULL, int numBlocks = 0)
            : algo(a),
              nBlocks(numBlocks),
              goal(std::numeric_limits<double>::max()),
              accuracyThreshold(0.0),
              tm(numBlocks),
              readInput(*this, "files", tm.tuner),
              step1Input1(*this, "step1Input1", tm.s1i1_tuner),
              step1Input2(*this, "step1Input2", tm.s1i2_tuner),
              result(*this, "result"),
              mr_graph(NULL),
              reader(NULL)
        {
            // ctrl_finalizer.prescribes(finalizer, *this);
            // finalizer.consumes(step2InOut);
            // finalizer.produces(result);
            // finalizer.produces(step1Input2);
            // finalizer.controls(ctrl_finalizer);
            
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this);
            if(algo) {
                accuracyThreshold = use_default(algo->_accuracyThreshold)
                    ? typename Algo::algob_type::ParameterType(algo->_nClusters, algo->_maxIterations).accuracyThreshold
                    : algo->_accuracyThreshold;
                // the graph instantiations must go last to have the same sequence as on remote processes (see serialize)
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
                mr_graph = new mr_graph_type(*this, a, numBlocks, step1Input1, step1Input2, result, &tm);
            }
        }
        
        ~mapRedIterContext()
        {
            delete mr_graph;
            delete reader;
        }
        
#ifdef _DIST_
        void serialize(CnC::serializer & ser)
        {
            if(ser.is_unpacking()) {
                assert(algo == NULL);
                algo = new Algo;
                delete mr_graph;
                delete reader;
            }
            ser & (*algo) & nBlocks & goal & accuracyThreshold & tm;
            if(ser.is_unpacking()) {
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
                mr_graph = new mr_graph_type(*this, algo, nBlocks, step1Input1, step1Input2, result, &tm);
            }
            else if(ser.is_cleaning_up()) delete algo;
        }
#endif

    };
    
    template< typename Algo >
    int mapRedIterContext< Algo >::Fini::execute(const int tag, mapRedIterContext< Algo > & ctxt) const
    {
        typedef typename Algo::iomstep2Master_type::result_type pres_type;
        typedef typename Algo::iomstep2Master__final_type::result_type res_type;
        
        pres_type pres;
        ctxt.mr_graph->step2InOut.get({ctxt.mr_graph->maxDepth, 0, tag}, pres);
        res_type res = ctxt.algo->run_step2Master__final(std::vector< typename Algo::iomstep2Master__final_type::input1_type >(1, pres));
            
        if(tag < ctxt.algo->_maxIterations) {
            double new_goal = res->get(daal::algorithms::kmeans::goalFunction)->daal::data_management::NumericTable::getValue<double>(0, 0);
            
            if(std::abs(ctxt.goal - new_goal) > ctxt.accuracyThreshold) {
                ctxt.step1Input2.put({0, tag+1}, res->get(daal::algorithms::kmeans::centroids));
                ctxt.mr_graph->ctrl_finalizer.put(tag+1);
                ctxt.goal = new_goal;
                return 0;
            }
        }

        // we get here it means we reached accuracy threshold or maxit: we are done!
        daal::data_management::NumericTablePtr maxittab(new daal::data_management::HomogenNumericTable<int>(1, 1, daal::data_management::NumericTable::doAllocate, (int)tag));
        res->set(daal::algorithms::kmeans::nIterations, maxittab);
        ctxt.result.put(0, res);
        return 0;
    }


    template< typename Algo >
    struct mapReduceIter
    {
        typedef Algo manager_type;
        typedef mapRedIterContext< Algo > context_type;

        static typename Algo::iomstep2Master__final_type::result_type
        compute(const TableOrFList & input, const TableOrFList & input2, Algo & algo)
        {
            algo._assignFlag = false;
            int pid = CnC::tuner_base::myPid();
            int nblocks = 1;
            if(! (input.table || input.file.size())) {
                nblocks = input.flist.size() ? input.flist.size() : input.tlist.size();
            }
            int offset = nblocks * CnC::tuner_base::myPid();

            context_type ctxt(&algo, nblocks*(CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1));
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(pid == 0) {
                assert(input2.table);
                ctxt.step1Input2.put({0, 0}, input2.table);
            }
            if(input.table) {
                ctxt.step1Input1.put({offset, 0}, input.table);
            } else if(input.file.size()) {
                ctxt.readInput.put({offset, 0}, {input.file});
            } else {
                assert((input.flist.size() == 0) != (input.tlist.size() == 0));
                for(int i = 0; i < input.flist.size(); ++i) {
                    ctxt.readInput.put({offset + i, 0}, {input.flist[i]});
                }
                for(int i = 0; i < input.tlist.size(); ++i) {
                    ctxt.step1Input1.put({offset + i, 0}, input.tlist[i]);
                }
            }
            ctxt.wait();
            
            typename Algo::iomstep2Master__final_type::result_type res;
            ctxt.result.get(0, res);

            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }
    };

} // namespace mapReduceIter


#endif // _MAP_RED_ITER_INCLUDED_
