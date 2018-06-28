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

#ifndef _APPLY_GATHER_INCLUDED_
#define _APPLY_GATHER_INCLUDED_

#include "cnc4daal.h"
#include "readGraph.h"
#include <cstdlib>

namespace applyGather {

    template< typename Algo > struct context;

    template< int NI >
    struct step1_default {
        template< typename Context >
        int execute( const int &, Context & ) const;
    };

    struct step2_default {
        template< typename Context >
        int execute( const int &, Context & ) const;
    };

    struct applyTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::tag_tuner<>
    {
        template< typename Arg >
        int compute_on( const int tag, Arg & /*arg*/ ) const
        {
            return tag % tuner_base::numProcs();
        }

        int consumed_on( const int tag ) const
        {
            return tag % tuner_base::numProcs();
        }

        int get_count( const int ) const
        {
            return 1;
        }
    };

    template< int P >
    struct gatherTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::preserve_tuner<int>
    {
        template< typename Arg >
        int compute_on( const int, Arg & /*arg*/ ) const
        {
            return 0;
        }

        int consumed_on( const int ) const
        {
            return P;
        }

        int get_count( const int ) const
        {
            return 1;
        }
    };

    template< typename Algo, typename step1=step1_default<1>, typename step2=step2_default >
    class applyGatherContext : public CnC::context< applyGatherContext< Algo, step1, step2 > >
    {
    public:
        typedef Algo algo_type;
        typedef CnC::identityMap< int > step1_map;
        typedef CnC::singletonMap< int, 0 > step2_map;
        typedef readGraph< int, applyTuner, applyTuner, applyTuner, Algo::NI > reader_type;
        typedef typename reader_type::in_coll_type read_input_coll_type;

        Algo * algo;
        int nBlocks;

        read_input_coll_type readInput;
        reader_type * reader;

        CnC::dc_step_collection< int, int, step1_map, step1, applyTuner > step_1;
        CnC::dc_step_collection< int, int, step2_map, step2, gatherTuner<0> > step_2;

        CnC::item_collection< int, typename Algo::iomstep1Local_type::input1_type,  applyTuner > step1Input1;
        CnC::item_collection< int, typename Algo::iomstep1Local_type::input1_type,  applyTuner > step1Input2;
        CnC::item_collection< int, typename Algo::iomstep2Master_type::input1_type, gatherTuner<0> > step2Input;
        CnC::item_collection< int, typename Algo::iomstep2Master_type::result_type, gatherTuner<CnC::CONSUMER_ALL> > result;

        applyGatherContext(Algo * a = NULL, int numBlocks = 0)
            : algo(a),
              nBlocks(numBlocks),
              readInput(*this, "files"),
              reader(NULL),
              step_1( *this, "apply" ),
              step_2( *this, "gather" ),
              step1Input1(*this, "step1Input1"),
              step1Input2(*this, "step1Input2"),
              step2Input(*this, "step2Input"),
              result( *this, "result")
        {
            step_1.consumes(step1Input1, step1_map());
            if(Algo::NI > 1) step_1.consumes(step1Input2);
            step_1.produces( step2Input );
            step_2.consumes( step2Input, step2_map() );
            step_2.produces( result );
            reader = new reader_type(*this, readInput, step1Input1, step1Input2);
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this);
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
            ser & (*algo) & nBlocks;
            if(ser.is_cleaning_up()) delete algo;
        }
#endif

    };

    template<>
    template< typename Context >
    int step1_default< 1 >::execute(const int & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input1_type pData;
        ctxt.step1Input1.get(tag, pData);

        typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData);

        ctxt.step2Input.put(tag, res);
        return 0;
    }

    template<>
    template< typename Context >
    int step1_default< 2 >::execute(const int & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input1_type pData1;
        ctxt.step1Input1.get(tag, pData1);
        typename Context::algo_type::iomstep1Local_type::input2_type pData2;
        ctxt.step1Input2.get(tag, pData2);

        typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData1, pData2);

        ctxt.step2Input.put(tag, res);
        return 0;
    }

    template< typename Context >
    int step2_default::execute(const int & tag, Context & ctxt) const
    {
        std::vector< typename Context::algo_type::iomstep2Master_type::input1_type > inp(ctxt.nBlocks);
        for( int i = 0; i < ctxt.nBlocks ; i++ ) {
            ctxt.step2Input.get(i, inp[i]);
        }

        typename Context::algo_type::iomstep2Master_type::result_type res = ctxt.algo->run_step2Master(inp);

        ctxt.result.put(tag, res);
        return 0;
    }

    template< typename Algo, typename step1=step1_default< Algo::NI >, typename step2=step2_default >
    struct applyGather
    {
        typedef Algo manager_type;
        typedef applyGatherContext< Algo, step1, step2 > context_type;

        static typename Algo::iomstep2Master_type::result_type
        compute(const TableOrFList & input, Algo & algo)
        {
            int pid = CnC::tuner_base::myPid();
            int stride = CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1;
            int nblocks = stride;
            if( ! (input.table || input.file.size()) ) {
                nblocks *= input.flist.size() ? input.flist.size() : input.tlist.size();
            }

            context_type ctxt(&algo, nblocks);
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input.table) {
                ctxt.step1Input1.put(pid, input.table);
            } else if(input.file.size()) {
                ctxt.readInput.put(pid, {input.file});
            } else {
                assert((input.flist.size() == 0) != (input.tlist.size() == 0));
                for(int i = 0; i < input.flist.size(); ++i) {
                    ctxt.readInput.put(pid + i * stride, {input.flist[i]});
                }
                for(int i = 0; i < input.tlist.size(); ++i) {
                    ctxt.step1Input1.put(pid + i * stride, input.tlist[i]);
                }
            }
            ctxt.wait();
            typename Algo::iomstep2Master_type::result_type res;
            ctxt.result.get(0, res);

            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }

        static typename Algo::iomstep2Master_type::result_type
        compute(const TableOrFList & input1, const TableOrFList & input2, Algo & algo)
        {
            int pid = CnC::tuner_base::myPid();
            int stride = CnC::Internal::distributor::distributed_env() ? CnC::tuner_base::numProcs() : 1;
            int nblocks = stride;
            if( ! (input1.table || input1.file.size()) ) {
                nblocks *= input1.flist.size() ? input1.flist.size() : input1.tlist.size();
            }

            context_type ctxt(&algo, nblocks);
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input1.table) {
                ctxt.step1Input1.put(pid, input1.table);
                ctxt.step1Input2.put(pid, input2.table);
            } else if(input1.file.size()) {
                assert(input2.file.size());
                ctxt.readInput.put(pid, {input1.file, input2.file});
            } else {
                assert(input1.flist.size() == input2.flist.size());
                assert(input1.tlist.size() == input1.tlist.size());
                assert((input1.flist.size() == 0) != (input1.tlist.size() == 0));
                for(int i = 0; i < input1.flist.size(); ++i) {
                    ctxt.readInput.put(pid + i * stride, {input1.flist[i], input2.flist[i]});
                }
                for(int i = 0; i < input1.tlist.size(); ++i) {
                    ctxt.step1Input1.put(pid + i * stride, input1.tlist[i]);
                    ctxt.step1Input2.put(pid + i * stride, input2.tlist[i]);
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

} // namespace applyGather


#endif // _APPLY_GATHER_INCLUDED_
