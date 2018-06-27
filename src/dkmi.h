/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#ifndef _KMEANS_DISTR_H_INCLUDED_
#define _KMEANS_DISTR_H_INCLUDED_

#include "daal.h"
#include "cnc4daal.h"
#include <array>
#include <numeric>

template< typename fptype, algorithms::kmeans::init::Method method >
struct kmeans_init_manager;

namespace dkmi
{

    typedef int it_tag;
    typedef int block_tag;
    typedef std::array< int, 2 > itBlock_tag;

    typedef daal::data_management::NumericTablePtr step2Input_t;
    typedef daal::data_management::DataCollectionPtr localNodeData_t;
    typedef daal::data_management::NumericTablePtr step2Result_t;
    typedef daal::data_management::NumericTablePtr step4Input_t;
    typedef daal::data_management::NumericTablePtr block_t;

    struct step2Ctrl {
        step2Ctrl(int nb=0, int nc=0) : m_nBlocks(nb), m_nClusters(nc) {}
        std::vector<itBlock_tag> operator()(const it_tag & tag) {
            std::vector<itBlock_tag> v(m_nBlocks);
            if(tag < m_nClusters) for(int i = 0; i<m_nBlocks; ++i) v[i] = {tag, i};
            return tag >= m_nClusters ? std::vector<itBlock_tag>(0) : v;
        }
    private:
        int m_nBlocks, m_nClusters;
    };

    struct step3Ctrl {
        std::vector< it_tag > operator()(const itBlock_tag & tag)
        {
            return tag[1] == 0 ? std::vector< it_tag >(1, tag[0]) : std::vector< it_tag >(0);
        }
    };


    struct step1 {
        template< typename Context >
        int execute( const block_tag &, Context & ) const;
    };
    struct step2 {
        template< typename Context >
        int execute( const itBlock_tag &, Context & ) const;
    };
    struct step3 {
        template< typename Context >
        int execute( const it_tag &, Context & ) const;
    };
    struct step4 {
        template< typename Context >
        int execute( const itBlock_tag &, Context & ) const;
    };


    struct localTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::tag_tuner<>
    {
        template< typename Arg >
        int compute_on( const block_tag tag, Arg & /*arg*/ ) const
        {
            return tag % tuner_base::numProcs();
        }

        int consumed_on( const block_tag tag ) const
        {
            return tag % tuner_base::numProcs();
        }

        template< typename Arg >
        int compute_on( const itBlock_tag tag, Arg & /*arg*/ ) const
        {
            return tag[1] % tuner_base::numProcs();
        }

        int consumed_on( const itBlock_tag tag ) const
        {
            return tag[1] % tuner_base::numProcs();
        }

        int get_count( const int ) const
        {
            return CnC::NO_GETCOUNT;
        }

        int get_count( const itBlock_tag ) const
        {
            return 1;
        }
    };

    struct step2Tuner : public CnC::step_tuner<>, public CnC::preserve_tuner< itBlock_tag >
    {
        template< typename Arg >
        int compute_on( const itBlock_tag tag, Arg & /*arg*/ ) const
        {
            return tag[1] % tuner_base::numProcs();
        }
    };

    struct s2iTuner : public CnC::hashmap_tuner
    {
        int consumed_on( const it_tag tag ) const
        {
            return CnC::CONSUMER_ALL;
        }
    };

    struct lDataTuner : public CnC::hashmap_tuner
    {
        int consumed_on( const itBlock_tag tag ) const
        {
            return tag[1] % tuner_base::numProcs();
        }
        int get_count( const itBlock_tag ) const
        {
            return 2;
        }
    };

    struct masterTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::preserve_tuner<int>
    {
        template< typename Arg >
        int compute_on( const it_tag, Arg & /*arg*/ ) const
        {
            return 0;
        }

        int consumed_on( const itBlock_tag ) const
        {
            return 0;
        }

        int get_count( const itBlock_tag ) const
        {
            return 1;
        }
    };

    template< typename Manager >
    struct dkmiContext : public CnC::context< dkmiContext< Manager > >
    {};

    template<typename fptype>
    struct dkmiContext< kmeans_init_manager< fptype, algorithms::kmeans::init::plusPlusDense > >
        : public CnC::context< dkmiContext< kmeans_init_manager< fptype, algorithms::kmeans::init::plusPlusDense > > >
    {
        typedef CnC::identityMap< block_tag >   step1_map;
        typedef step2Ctrl                       step2_map;
        typedef step3Ctrl                       step3_map;
        typedef CnC::identityMap< itBlock_tag > step4_map;

        typedef kmeans_init_manager< fptype, algorithms::kmeans::init::plusPlusDense > manager_type;
        typedef readGraph< block_tag, localTuner, localTuner, localTuner, 1 > reader_type;
        typedef typename reader_type::in_coll_type read_input_coll_type;

        manager_type * algo;
        int nBlocks;
        read_input_coll_type readInput;
        reader_type * reader;
        
        // Step collections
        CnC::dc_step_collection< block_tag,   block_tag,   step1_map, step1, localTuner > step_1;
        CnC::dc_step_collection< it_tag,      itBlock_tag, step2_map, step2, step2Tuner > step_2;
        CnC::dc_step_collection< itBlock_tag, it_tag,      step3_map, step3, masterTuner > step_3;
        CnC::dc_step_collection< itBlock_tag, itBlock_tag, step4_map, step4, localTuner > step_4;

        // Item collections
        CnC::item_collection< block_tag, block_t, localTuner > step1Input1; //pData in examples
        CnC::item_collection< block_tag, block_t, localTuner > step1Input2; //dummy
        CnC::item_collection< it_tag,      step2Input_t,    s2iTuner > step2Input;
        CnC::item_collection< itBlock_tag, localNodeData_t, lDataTuner > localNodeData;
        CnC::item_collection< itBlock_tag, step2Result_t,   masterTuner > step2Result;
        CnC::item_collection< itBlock_tag, step4Input_t,    localTuner > step4Input;


        // The context class constructor
        dkmiContext(manager_type * a = NULL, int nb = 0)
            : algo(a),
              nBlocks(nb),
              readInput(*this, "files"),
              reader(NULL),
              step_1( *this, "dkmi-step1" ),
              step_2( *this, "dkmi-step2" ),
              step_3( *this, "dkmi-step3" ),
              step_4( *this, "dkmi-step4" ),
              step1Input1(*this, "dkmi-step1Input1"),
              step1Input2(*this, "dkmi-dummy"),
              step2Input(*this, "dkmi-step2Input"),
              localNodeData( *this, "dkmi-localNodeData" ),
              step2Result( *this, "dkmi-step2Result" ),
              step4Input( *this, "dkmi-step4Input" )
        {
            step_1.consumes( step1Input1, step1_map() );
            step_1.produces( step2Input );

            step_2.consumes( step1Input1 );
            step_2.consumes( localNodeData );
            step_2.produces( localNodeData );
            step_2.produces( step2Result );

            step_3.consumes( step2Result, step3_map() );
            step_3.produces( step4Input );

            step_4.consumes( step4Input, step4_map() );
            step_4.consumes( step1Input1 );
            step_4.consumes( localNodeData );
            step_4.produces( step2Input );
            if(a) {
                step_2.consumes( step2Input, step2_map(nBlocks, a->_nClusters) );
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
            }
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this);
        }
        ~dkmiContext()
        {
            delete reader;
        }

#ifdef _DIST_
        virtual void serialize(CnC::serializer & ser)
        {
            if(ser.is_unpacking()) {
                assert(algo == NULL);
                algo = new manager_type;
            }
            ser & (*algo) & nBlocks;
            if( ser.is_unpacking() ) {
                step_2.consumes( step2Input, step2_map(nBlocks, algo->_nClusters) );
                reader = new reader_type(*this, readInput, step1Input1, step1Input2);
            }
            else if(ser.is_cleaning_up()) delete algo;
        }
#endif
    };


    template< typename Context >
    int step1::execute( const block_tag & tag, Context & ctxt ) const
    {
        block_t pData;
        ctxt.step1Input1.get(tag, pData);

        typename Context::manager_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData, pData->getNumberOfRows()*ctxt.nBlocks, pData->getNumberOfRows()*tag);

        if(res) {
            step2Input_t c = res->get(algorithms::kmeans::init::partialCentroids);
            if(c) ctxt.step2Input.put(1, c);
        }
        return 0;
    }

    template< typename Context >
    int step2::execute( const itBlock_tag & tag, Context & ctxt ) const
    {
        block_t pData;
        ctxt.step1Input1.get(tag[1], pData);
        step2Input_t step2Input;
        ctxt.step2Input.get(tag[0], step2Input);
        localNodeData_t localNodeData;
        if( tag[0] > 1 ) ctxt.localNodeData.get({tag[0]-1, tag[1]}, localNodeData);

        typename Context::manager_type::iomstep2Local_type::result_type res = ctxt.algo->run_step2Local(pData, localNodeData, step2Input);

        ctxt.step2Result.put(tag, res->get(algorithms::kmeans::init::outputOfStep2ForStep3));
        if(tag[0] == 1) localNodeData = res->get(algorithms::kmeans::init::internalResult);
        ctxt.localNodeData.put(tag, localNodeData);
        return 0;
    }

    template< typename Context >
    int step3::execute( const it_tag & tag, Context & ctxt ) const
    {
        std::vector< typename Context::manager_type::iomstep3Master_type::input1_type > inp(ctxt.nBlocks);
        for( int i = 0; i < ctxt.nBlocks ; i++ ) {
            ctxt.step2Result.get({tag, i}, inp[i]);
        }

        typename Context::manager_type::iomstep3Master_type::result_type res = ctxt.algo->run_step3Master(inp);

        for(int i = 0; i < ctxt.nBlocks; ++i) {
            daal::data_management::NumericTablePtr pTbl = res->get(algorithms::kmeans::init::outputOfStep3ForStep4, i); /* can be null */
            if( pTbl ) {
                ctxt.step4Input.put({tag, i}, pTbl);
                return 0;
            }
        }
        throw std::invalid_argument("No output from step3");
        return 0;
    }

    template< typename Context >
    int step4::execute( const itBlock_tag & tag, Context & ctxt ) const
    {
        step4Input_t step4Input;
        ctxt.step4Input.get(tag, step4Input);
        localNodeData_t localNodeData;
        ctxt.localNodeData.get(tag, localNodeData);
        block_t pData;
        ctxt.step1Input1.get(tag[1], pData);

        typename Context::manager_type::iomstep4Local_type::result_type res = ctxt.algo->run_step4Local(pData, localNodeData, step4Input);

        ctxt.step2Input.put(tag[0]+1, res);
        return 0;
    }

    struct step1ApplyGather {
        template< typename Context >
        int execute(const int & tag, Context & ctxt) const
        {
            typename Context::algo_type::iomstep1Local_type::input1_type pData;
            ctxt.step1Input1.get(tag, pData);

            typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData, pData->getNumberOfRows()*ctxt.nBlocks, pData->getNumberOfRows()*tag);

            ctxt.step2Input.put(tag, res);
            return 0;
        }
    };

    template< typename Manager >
    struct dkmi
    {
        typedef Manager manager_type;
        typedef dkmiContext< manager_type > context_type;

        static typename manager_type::iomb_type::result_type
        compute(const TableOrFList & input, Manager & algo)
        {
            throw std::logic_error("Distributed kmeans-init not implemented for given configuration");
        }
    };

    // template< typename Manager >
    // typename Manager::iomstep2Master_type::result_type
    // dkmi_ag(const TableOrFList & input, Manager & algo)
    // {
    //     return applyGather::applyGather< Manager, step1ApplyGather >::compute(input, algo);
    // }

    template< typename Manager >
    struct dkmi_ag
    {
        typedef Manager manager_type;
        typedef typename applyGather::applyGather< manager_type, step1ApplyGather, applyGather::step2_default >::context_type context_type;

        static typename manager_type::iomb_type::result_type
        compute(const TableOrFList & input, manager_type & algo)
        {
            auto pCentroids = applyGather::applyGather< manager_type, step1ApplyGather, applyGather::step2_default >::compute(input, algo);
            typename manager_type::iomb_type::result_type res(new typename manager_type::iomb_type::result_type::ElementType);
            if(CnC::tuner_base::myPid() == 0 || CnC::Internal::distributor::distributed_env()) {
                res->set(algorithms::kmeans::init::centroids, pCentroids);
            }
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }
    };

    template< typename fptype >
    struct dkmi< kmeans_init_manager< fptype, algorithms::kmeans::init::randomDense > >
        : public dkmi_ag< kmeans_init_manager< fptype, algorithms::kmeans::init::randomDense > >
    {};
    template< typename fptype >
    struct dkmi< kmeans_init_manager< fptype, algorithms::kmeans::init::deterministicDense > >
        : public dkmi_ag< kmeans_init_manager< fptype, algorithms::kmeans::init::deterministicDense > >
    {};

    template<typename fptype>
    struct dkmi< kmeans_init_manager< fptype, algorithms::kmeans::init::plusPlusDense > >
    {
        typedef kmeans_init_manager< fptype, algorithms::kmeans::init::plusPlusDense > manager_type;
        typedef dkmiContext< manager_type > context_type;

        static typename manager_type::iomb_type::result_type
        compute(const TableOrFList & input, manager_type & algo)
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
                ctxt.step1Input1.put(CnC::tuner_base::myPid(), input.table);
            } else if(input.file.size()) {
                ctxt.readInput.put(CnC::tuner_base::myPid(), {input.file});
            } else {
                assert((input.flist.size() == 0) || (input.tlist.size() == 0));
                for(int i = 0; i < input.tlist.size(); ++i) {
                    ctxt.step1Input1.put(pid + i * stride, input.tlist[i]);
                }
                for(int i = 0; i < input.flist.size(); ++i) {
                    ctxt.readInput.put(pid + i * stride, {input.flist[i]});
                }
            }
            ctxt.wait();
            typename manager_type::iomb_type::result_type res(new typename manager_type::iomb_type::result_type::ElementType);
            data_management::RowMergedNumericTable pCentroids;
            
            for(int i = 1; i <= algo._nClusters; ++i ) {
                step2Input_t s2i;
                ctxt.step2Input.get(i, s2i);
                pCentroids.addNumericTable(s2i);
            }
            res->set(algorithms::kmeans::init::centroids, data_management::convertToHomogen<double>(pCentroids));
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            return res;
        }
    };

} // namespace kmeans_init

#endif // _KMEANS_DISTR_H_INCLUDED_
