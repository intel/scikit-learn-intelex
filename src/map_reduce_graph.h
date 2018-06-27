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

  We allow re-using the graph ad extending the tag. An template argument N
  is provided to add more tag-components for example to allow being used in a loop.
  In other words. it allows you to have multiple map-reduces in the same graph,
  potentially even at the same time. The extra tag-components are simply copied through
  the control-flow.
 */

#ifndef _MAP_REDUCE_GRAPH_INCLUDED_
#define _MAP_REDUCE_GRAPH_INCLUDED_

#include "cnc4daal.h"
#include <cstdlib>
#include <type_traits>

namespace mapReduce {
    typedef std::array<int, 2> nodeId_type;
    typedef std::pair<nodeId_type, nodeId_type> nodeIdPair_type;

    static int get_power2(size_t x)
    {
        int power = 1;
        while(power < x) power*=2;
        return power;
    }

    template< size_t N >
    static inline nodeIdPair_type get_children(const std::array< int, N > & id)
    {
        int cl = std::get<0>(id)-1;
        return id[0] > 0
            ? nodeIdPair_type({cl, std::get<1>(id)*2}, {cl, std::get<1>(id)*2+1})
            : nodeIdPair_type({-1, -1}, {-1, -1});
    }

    static inline nodeId_type get_parent(const nodeId_type & id)
    {
        return {id[0]+1, (int)(id[1]/2)};
    }

    static inline int get_compute_on(const nodeId_type & id, int nLeafs, int P)
    {
        if(id[0] <= 0) {
            int npr = nLeafs / P;
            return (int)(id[1] / npr) % P;
        }
        return get_compute_on(get_children(id).first, nLeafs, P);
    }

    static inline int get_consumed_on(const nodeId_type & id, int nLeafs, int P)
    {
        if(id[0] == -1) {
            int npr = nLeafs / P;
            return id[1] / npr;
        }
        return get_compute_on(get_parent(id), nLeafs, P);
    }

    template< size_t N >
    struct tagger
    {
        typedef std::array< int, N+1 > step1_tag_type;
        typedef std::array< int, N+2 > step2_tag_type;
    };

    template< size_t N >
    typename tagger< N >::step2_tag_type step2_tag(const int d, const int p, typename tagger< N >::step2_tag_type o)
    {
        std::get<0>(o) = d;
        std::get<1>(o) = p;
        return o;
    }

    template< size_t N >
    typename tagger< N >::step2_tag_type step2_tag(const nodeId_type & i, typename tagger< N >::step2_tag_type o)
    {
        std::get<0>(o) = std::get<0>(i);
        std::get<1>(o) = std::get<1>(i);
        return o;
    }

    template< size_t N >
    typename tagger< N >::step2_tag_type step2_tag(const typename tagger< N >::step1_tag_type & o)
    {
        typename tagger< N >::step2_tag_type rv;
        std::get<0>(rv) = 0;
        std::copy(o.begin(), o.end(), rv.begin()+1);
        return rv;
    }

    template< size_t N >
    nodeId_type nodeId(const typename tagger< N >::step2_tag_type & tag)
    {
        return {std::get<0>(tag), std::get<1>(tag)};
    }
    template< size_t N >
    nodeId_type nodeId(const typename tagger< N >::step1_tag_type tag)
    {
        return {0, std::get<0>(tag)};
    }
    template< size_t N >
    nodeId_type nodeId(int d, const typename tagger< N >::step1_tag_type & tag)
    {
        //static_assert(N>0, "Should use specialized template function");
        return {d, std::get<0>(tag)};
    }

    template< size_t N >
    struct mapRedTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::tag_tuner<>
    {
        int n2Blocks;

        mapRedTuner(int nb = 0) : n2Blocks(nb > 0 ? get_power2(nb) : -1) {}

        // this is for step2 on inner nodes
        template< typename Arg >
        int compute_on(const typename tagger< N >::step2_tag_type & tag, Arg & /*arg*/) const
        {
            return get_compute_on(nodeId<N>(tag), n2Blocks, tuner_base::numProcs());
        }
        // this is for step1 on leafs
        template< typename Arg >
        int compute_on(const typename tagger< N >::step1_tag_type & tag, Arg & /*arg*/) const
        {
            return get_compute_on(nodeId<N>(tag), n2Blocks, tuner_base::numProcs());
        }

        // this is for step2 on inner nodes
        int consumed_on(const typename tagger< N >::step2_tag_type & tag) const
        {
            return get_consumed_on(nodeId<N>(tag), n2Blocks, tuner_base::numProcs());
        }
        // this is for step1 on leafs
        int consumed_on(const typename tagger< N >::step1_tag_type & tag) const
        {
            return get_consumed_on(nodeId<N>(-1, tag), n2Blocks, tuner_base::numProcs());
        }

        template< typename T >
        int get_count(const T &) const
        {
            return 1;
        }
    };

    
    struct localTuner : public CnC::step_tuner<>, public CnC::vector_tuner, public CnC::tag_tuner<>
    {
        template< typename T, typename Arg >
        int compute_on(const T&, Arg &) const
        {
            return CnC::COMPUTE_ON_LOCAL;
        }
        template< typename T >
        int consumed_on(const T&) const
        {
            return CnC::CONSUMER_ALL;
        }
        template< typename T >
        int get_count(const T&) const
        {
            return 1;
        }
    };

    
    struct tag_manager
    {
        tag_manager(int nb=-1) : tuner(nb) {}
        
        static const size_t N = 0;

        typedef tagger<N> tagger_type;
        typedef mapRedTuner<N> s1i1_tuner_type;
        typedef mapRedTuner<N> s1i2_tuner_type;
        typedef mapRedTuner<N> tuner_type;
        tuner_type tuner;
        
        typedef CnC::identityMap< tagger< N >::step1_tag_type > step1_map_type;
        step1_map_type s1i1_map;
        CnC::no_mapper s1i2_map;
        

        template< typename T >
        static inline T input_s1(const int, const T & t) {return t;}
    };

#ifdef _DIST_
    CNC_BITWISE_SERIALIZABLE(mapRedTuner<0>);
    CNC_BITWISE_SERIALIZABLE(mapRedTuner<1>);
    CNC_BITWISE_SERIALIZABLE(localTuner);
    CNC_BITWISE_SERIALIZABLE(tag_manager);
#endif
    
    template< typename Ctxt, typename TM=tag_manager >
    class mapReduceGraph : public CnC::graph
    {
    public:
        typedef tagger< TM::N > tagger_type;
        typedef TM tag_mgr_type;
        typedef typename tagger_type::step1_tag_type step1_tag_type;
        typedef typename tagger_type::step2_tag_type step2_tag_type;
        typedef CnC::item_collection< step1_tag_type, data_management::NumericTablePtr, typename TM::s1i1_tuner_type > step1_input1_coll_type;
        typedef CnC::item_collection< step1_tag_type, data_management::NumericTablePtr, typename TM::s1i2_tuner_type > step1_input2_coll_type;
        typedef CnC::item_collection< step2_tag_type, typename Ctxt::algo_type::iomstep2Master_type::input1_type, typename TM::tuner_type > step2_input_coll_type;
        typedef CnC::tag_collection< int, localTuner > finalizer_tag_coll_type;
        typedef CnC::step_collection< typename Ctxt::Fini, localTuner > finalizer_step_coll_type;
        typedef CnC::item_collection< int, typename Ctxt::algo_type::iomstep2Master__final_type::result_type, localTuner > result_coll_type;

        struct step1 {
            int execute(const typename tagger_type::step1_tag_type &, mapReduceGraph< Ctxt, TM > &) const;
        };

        struct step2 {
            int execute(const typename tagger_type::step2_tag_type &, mapReduceGraph< Ctxt, TM > &) const;
        };

        typename Ctxt::algo_type * algo;
        int nBlocks;
        int maxDepth;
        TM tm;

        CnC::tag_collection< step2_tag_type, typename TM::tuner_type > ctrl_2;
        finalizer_tag_coll_type ctrl_finalizer;

        step1_input1_coll_type & step1Input1;
        step1_input2_coll_type & step1Input2;
        result_coll_type & result;
        step2_input_coll_type step2InOut;
        
        CnC::dc_step_collection< step1_tag_type, step1_tag_type, typename TM::step1_map_type, step1, typename TM::tuner_type > step_1;
        CnC::step_collection< step2, typename TM::tuner_type > step_2;
        finalizer_step_coll_type finalizer;

        mapReduceGraph(Ctxt & ctxt,
                       typename Ctxt::algo_type * a,
                       int numBlocks,
                       step1_input1_coll_type & s1i1,
                       step1_input2_coll_type & s1i2,
                       result_coll_type & res,
                       TM * _tm = NULL,
                       const std::string & name = "MapReduce")
            : CnC::graph(ctxt, name),
              algo(a),
              nBlocks(numBlocks),
              maxDepth(log2(mapReduce::get_power2(nBlocks))),
              tm(),
              ctrl_2(ctxt, "ctrl2", tm.tuner),
              ctrl_finalizer(ctxt, "ctrl_fini"),
              step1Input1(s1i1),
              step1Input2(s1i2),
              result(res),
              step2InOut(ctxt, "step2InOut", tm.tuner),
              step_1(ctxt, tm.tuner, "map", *this),
              step_2(ctxt, tm.tuner, "reduce"),
              finalizer(ctxt, "fini")
        {
            tm = _tm ? *_tm : TM(numBlocks);
            step_1.consumes(step1Input1, tm.s1i1_map);
            if( Ctxt::algo_type::NI > 1 ) step_1.consumes(step1Input2, tm.s1i2_map);
            step_1.produces(step2InOut);
            step_1.controls(ctrl_2);
            ctrl_2.prescribes(step_2, *this);
            step_2.consumes(step2InOut);
            step_2.produces(step2InOut); // result,
            ctrl_finalizer.prescribes(finalizer, ctxt);
            result.set_max(1);
            if(CnC::tuner_base::myPid() == 0) ctrl_finalizer.put(0);
        }

        void emit_ctrl(const step2_tag_type & tag)
        {
            // we need to prescribe all reduce-steps
            int p2 = get_power2(nBlocks);
            int depth = 0;
            while(p2 >= 1) {
                p2 /= 2;
                ++depth;
                for(int i=0; i<p2; ++i) {
                    ctrl_2.put(step2_tag< tag_mgr_type::N >(depth, i, tag));
                }
            }
        }


    };

    template< int N, typename TM >
    struct step1_impl
    {
        template< typename Context >
        static int execute(const typename Context::tagger_type::step1_tag_type & tag, Context & ctxt);
    };

    template<>
    template< typename TM >
    struct step1_impl< 1, TM >
    {
        template< typename Context >
        static int execute(const typename Context::tagger_type::step1_tag_type & tag, Context & ctxt)
        {
            typename Context::step1_input_coll_type::data_type pData;
            ctxt.step1Input1.get(tag, pData);

            typename Context::step2_input_coll_type::data_type res = ctxt.algo->run_step1Local(pData);

            typename Context::taggertype::step2_tag_type otag = step2_tag< TM::N >(tag);
            ctxt.step2InOut.put(otag, res);

            if(std::get<0>(otag) == 0 && std::get<1>(otag) == 0) {
                ctxt.emit_ctrl(otag);
            }

            return 0;
        }
    };

    template<>
    template< typename TM >
    struct step1_impl< 2, TM >
    {
        template< typename Ctxt >
        static int execute(const typename Ctxt::tagger_type::step1_tag_type & tag, Ctxt & ctxt)
        {
            typename Ctxt::step1_input1_coll_type::data_type pData1;
            ctxt.step1Input1.get(Ctxt::tag_mgr_type::input_s1(0, tag), pData1);
            typename Ctxt::step1_input2_coll_type::data_type pData2;
            ctxt.step1Input2.get(Ctxt::tag_mgr_type::input_s1(1, tag), pData2);

            typename Ctxt::step2_input_coll_type::data_type res = ctxt.algo->run_step1Local(pData1, pData2);

            typename Ctxt::tagger_type::step2_tag_type otag = step2_tag< TM::N >(tag);
            ctxt.step2InOut.put(otag, res);

            if(std::get<0>(tag) == 0) {
                ctxt.emit_ctrl(otag);
            }

            return 0;
        }
    };

    template< typename Ctxt, typename TM >
    int mapReduceGraph< Ctxt, TM >::step1::execute(const typename mapReduceGraph< Ctxt, TM >::tagger_type::step1_tag_type & tag,
                                                   mapReduceGraph< Ctxt, TM > & ctxt) const
    {
        return step1_impl< Ctxt::algo_type::NI, TM >::execute(tag, ctxt);
    }

    template< typename Ctxt, typename TM >
    int mapReduceGraph< Ctxt, TM >::step2::execute(const typename mapReduceGraph< Ctxt, TM >::tagger_type::step2_tag_type & tag,
                                                   mapReduceGraph< Ctxt, TM > & ctxt) const
    {
        nodeIdPair_type c = get_children(tag);
        std::vector< typename mapReduceGraph< Ctxt, TM >::step2_input_coll_type::data_type > inp(2);
        if(std::get<1>(c.first) < ctxt.nBlocks) ctxt.step2InOut.get(step2_tag< TM::N >(c.first, tag), inp[0]);
        if(std::get<1>(c.second) < ctxt.nBlocks) ctxt.step2InOut.get(step2_tag< TM::N >(c.second, tag), inp[1]);

        typename mapReduceGraph< Ctxt, TM >::step2_input_coll_type::data_type res = ctxt.algo->run_step2Master(inp);

        ctxt.step2InOut.put(tag, res);

        return 0;
    }

} // namespace mapReduce


#endif // _MAP_REDUCE_GRAPH_INCLUDED_
