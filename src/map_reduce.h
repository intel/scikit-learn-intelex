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

/*
  We implement a binary tree reduction. The leafs of the tree are our input blocks.
  Each input block is mapped to step1 with block-id as control/data tags.
  The inner nodes are applied to step2 with 2 input nodes each.
  Tree nodes are identified by a pair (depth, pos)
    - depth is the level in the tree with 0 for leafs
    - pos left-to-right [0,n[ numbered position on the level
    - parent (d,x)'s children are (d-1,x/2) and (d-1,/2+1)
 */

#ifndef _MAP_REDUCE_INCLUDED_
#define _MAP_REDUCE_INCLUDED_

#include "cnc4daal.h"
#include "readGraph.h"
#include "map_reduce_graph.h"
#include <cstdlib>

namespace mapReduce {


    template< typename Algo > //, typename step1=step1_default<1>, typename step2=step2_default >
    class mapReduceContext : public CnC::context< mapReduceContext< Algo > > //, step1, step2
    {
    public:
        struct Fini {
            int execute(const int, mapReduceContext< Algo > &) const;
        };
        
        typedef Algo algo_type;
        typedef tag_manager tag_manager_type;
        typedef tag_manager_type::tagger_type::step1_tag_type step1_tag_type;
        typedef readGraph< step1_tag_type, mapRedTuner<0>, mapRedTuner<0>, mapRedTuner<0>, Algo::NI > reader_type;
        typedef typename reader_type::in_coll_type read_input_coll_type;

        typedef mapReduceGraph< mapReduceContext< Algo > > mr_graph_type;
        typedef typename mr_graph_type::step1_input1_coll_type step1_input1_coll_type;
        typedef typename mr_graph_type::step1_input2_coll_type step1_input2_coll_type;
        typedef typename mr_graph_type::step2_input_coll_type step2_input_coll_type;
        typedef typename mr_graph_type::result_coll_type result_coll_type;
        

        Algo * algo;
        int nBlocks;
        tag_manager tm;
        
        read_input_coll_type readInput;
        step1_input1_coll_type step1Input1;
        step1_input2_coll_type step1Input2;
        result_coll_type result;
        
        mr_graph_type * mr_graph;
        reader_type * reader;

        mapReduceContext(Algo * a = NULL, int numBlocks = 0)
            : algo(a),
              nBlocks(numBlocks),
              tm(numBlocks),
              readInput(*this, "files", tm.tuner),
              step1Input1(*this, "step1Input1", tm.tuner),
              step1Input2(*this, "step1Input2", tm.tuner),
              result(*this, "result"),
              mr_graph(NULL),
              reader(NULL)
        {
            if(a) {
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
                mr_graph = new mr_graph_type(*this, a, numBlocks, step1Input1, step1Input2, result, &tm);
            }
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this, 15);
        }
        
        ~mapReduceContext()
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
            ser & (*algo) & nBlocks & tm;
            if(ser.is_unpacking()) {
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
                mr_graph = new mr_graph_type(*this, algo, nBlocks, step1Input1, step1Input2, result, &tm);
            }
            else if(ser.is_cleaning_up()) delete algo;
        }
#endif

    };

    
    template< typename Algo >
    int mapReduceContext< Algo >::Fini::execute(const int tag, mapReduceContext< Algo > & ctxt) const
    {
        int depth = log2(ctxt.tm.tuner.n2Blocks);
        typename Algo::iomstep2Master_type::result_type pres;
        ctxt.mr_graph->step2InOut.get({depth, 0}, pres);
        typename Algo::iomstep2Master__final_type::result_type res;
        res = ctxt.algo->run_step2Master__final(std::vector< typename Algo::iomstep2Master_type::result_type >(1, pres));
        ctxt.result.put(0, res);
        return 0;
    }


    template< typename Algo > //, typename step1=step1_default< Algo::NI >, typename step2=step2_default >
    struct mapReduce
    {
        typedef Algo manager_type;
        typedef mapReduceContext< Algo/*, step1, step2*/ > context_type;

        static typename Algo::iomstep2Master__final_type::result_type
        compute(context_type & ctxt)
        {
            ctxt.wait();
            typename Algo::iomstep2Master__final_type::result_type res;
            if(CnC::tuner_base::myPid() == 0 || CnC::Internal::distributor::distributed_env()) {
                ctxt.result.get(0, res);
            }
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }
        
        static typename Algo::iomstep2Master__final_type::result_type
        compute(const TableOrFList & input, Algo & algo)
        {
            int nblocks = 1;
            if(! (input.table || input.file.size())) {
                nblocks = input.flist.size() ? input.flist.size() : input.tlist.size();
            }
            int offset = nblocks * CnC::tuner_base::myPid();

            context_type ctxt(&algo, nblocks*(CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1));
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input.table) {
                ctxt.step1Input1.put({offset}, input.table);
            } else if(input.file.size()) {
                ctxt.readInput.put({offset}, {input.file});
            } else {
                assert((input.flist.size() == 0) != (input.tlist.size() == 0));
                for(int i = 0; i < input.flist.size(); ++i) {
                    ctxt.readInput.put({offset + i}, {input.flist[i]});
                }
                for(int i = 0; i < input.tlist.size(); ++i) {
                    ctxt.step1Input1.put({offset + i}, input.tlist[i]);
                }
            }
            return compute(ctxt);
        }

        static typename Algo::iomstep2Master__final_type::result_type
        compute(const TableOrFList & input1, const TableOrFList & input2, Algo & algo)
        {
            int nblocks = 1;
            if(! (input1.table || input1.file.size())) {
                nblocks = input1.flist.size() ? input1.flist.size() : input1.tlist.size();
            }
            int offset = nblocks * CnC::tuner_base::myPid();

            context_type ctxt(&algo, nblocks*(CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1));
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input1.table) {
                ctxt.step1Input1.put({offset}, input1.table);
                ctxt.step1Input2.put({offset}, input2.table);
            } else if(input1.file.size()) {
                assert(input2.file.size());
                ctxt.readInput.put({offset}, {input1.file, input2.file});
            } else {
                assert(input1.flist.size() == input2.flist.size());
                assert(input1.tlist.size() == input1.tlist.size());
                assert((input1.flist.size() == 0) != (input1.tlist.size() == 0));
                for(int i = 0; i < input1.flist.size(); ++i) {
                    ctxt.readInput.put({offset + i}, {input1.flist[i], input2.flist[i]});
                }
                for(int i = 0; i < input1.tlist.size(); ++i) {
                    ctxt.step1Input1.put({offset + i}, input1.tlist[i]);
                    ctxt.step1Input2.put({offset + i}, input2.tlist[i]);
                }
            }
            return compute(ctxt);
        }
    };

} // namespace mapReduce


#endif // _MAP_REDUCE_INCLUDED_
