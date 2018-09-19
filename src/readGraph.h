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

#ifndef _READGRAPH_INCLUDED_
#define _READGRAPH_INCLUDED_

template< typename Tag, typename RTuner, typename OTuner1, typename OTuner2, size_t NI >
struct readGraph : public CnC::graph
{
    typedef std::array< std::string, NI > input_type;
    typedef CnC::item_collection< Tag, input_type, RTuner > in_coll_type;
    typedef CnC::item_collection< Tag, daal::data_management::NumericTablePtr, OTuner1 > out_coll1_type;
    typedef CnC::item_collection< Tag, daal::data_management::NumericTablePtr, OTuner2 > out_coll2_type;
    typedef CnC::identityMap< Tag > step0_map;

    struct step0 {
        template< typename Context >
        int execute( const Tag & tag, Context & ctxt) const
        {
            typename Context::input_type fname;
            ctxt.in_coll.get(tag, fname);

            ctxt.out_coll1.put(tag, readCSV(fname[0]));
            if(NI > 1) ctxt.out_coll2.put(tag, readCSV(fname[1]));
            
            return 0;
        }
    };

    in_coll_type & in_coll;
    out_coll1_type & out_coll1;
    out_coll2_type & out_coll2;
    CnC::dc_step_collection< Tag, Tag, step0_map, step0, RTuner > step_0;

    template< typename Ctxt >
    readGraph(CnC::context< Ctxt > & ctxt, in_coll_type & ic, out_coll1_type & oc1, out_coll2_type & oc2, const std::string & name = "reader")
        : CnC::graph(ctxt, name),
          out_coll1(oc1),
          out_coll2(oc2),
          step_0(ctxt, "read", *this),
          in_coll(ic)
    {
        step_0.consumes(ic, step0_map());
        step_0.produces( oc1 );
        if(NI > 1) step_0.produces( oc2 );
    }
};

#endif // _READGRAPH_INCLUDED_
