#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

###############################################################################
# The code generator.
# We define jinja2 templates to generate code for all the DAAL algorithms and their
# Result and Model objects. Most macros work on one namespace and expect expect
# values/variables in their env. If not specificied otherwise, we assume we can
# use the following
#          {{ns}}:            current C++ namespace
#          {{algo}}:          algo name (as seen in API)
#          {{args_decl}}:     non-template arguments for wrapper function
#          {{args_call}}:     non-template arguments to pass on
#          {{input_args}}:    args for setting input
#          {{template_decl}}: template parameters for declaration
#          {{template_args}}: template arguments mapping to their possible values
#          {{params_req}}:    dict of required parameters and their values
#          {{params_opt}}:    dict of parameters and their values
#          {{params_get}}:    parameter struct getter
#          {{step_specs}}:    distributed spec
#          {{map_result}}:    Result enum id for getting partial result (can be FULLPARTIAL)
#          {{iface}}:         Interface manager
#          {{result_map}}:    Type information about result type of given algo
#
# The code here is of course highly dependent on manual code that can be found in ../src.
# For example, we pretend we know how distributed algorithm interfaces look like and work.
#
# Also, we use some functions which accept PyObject* to convert the python objects manually
# Similarly, we create PyObjects manually in some C++ functions (e.g. nd-arrays).
###############################################################################
# FIXME remove remaining args/code for distributed computation if none available
# FIXME a revision from scratch would be helpful...
# FIXME GC of Tables etc (shared-pointer objcetss are new'ed!)

import jinja2
from collections import OrderedDict
from pprint import pprint
import re
from .wrappers import hpat_types

###############################################################################
# generic utility functions/defs needed by generated code
cython_header = '''
# distutils: language = c++

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as npc
# import std::string support
from libcpp.string cimport string as std_string
from libcpp cimport bool
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref

npc.import_array()

hpat_spec = []

cdef extern from "daal4py_cpp.h":
    cdef cppclass NumericTablePtr:
        pass

    cdef cppclass data_management_DataCollectionPtr:
        pass

    ctypedef NumericTablePtr data_management_NumericTablePtr


cdef class data_management_datacollection:
    cdef data_management_DataCollectionPtr * c_ptr

    def __dealloc__(self):
        del self.c_ptr


cdef extern from "daal4py.h":
    cdef const double NaN64
    cdef const float  NaN32

    cdef std_string to_std_string(PyObject * o) except +

    cdef PyObject * make_nda(NumericTablePtr * nt_ptr) except +
    cdef NumericTablePtr * make_nt(PyObject * nda) except +

    cdef cppclass TableOrFList :
        TableOrFList(PyObject *) except +
        pass


NAN64 = NaN64
NAN32 = NaN32


cdef extern from "daal.h":
    ctypedef unsigned long long DAAL_UINT64


cdef extern from "daal4py_cpp.h":
    cdef void c_daalinit(bool spmd, int flag) except +
    cdef void c_daalfini() except +
    cdef size_t c_num_procs() except +
    cdef size_t c_my_procid() except +


def daalinit(spmd = False, flag = 0):
    c_daalinit(spmd, flag)

def daalfini():
    c_daalfini()

def num_procs():
    return c_num_procs()

def my_procid():
    return c_my_procid()
'''

###############################################################################
# generates result/model classes
# Generates header, c++ and cython code together, separated by "%SNIP%
# requires {{enum_gets}}    list of triplets of members accessed via get(ns, name, type)
#          {{named_gets}}   list of pairs(name, type) of members accessed via type var = getName()
#          {{class_type}}   C++ class
typemap_wrapper_template = """
{% set flatname = (class_type|flat|strip(' *')|lower).replace('ptr', '') %}
{% set splitname = class_type.rsplit('::', 1) %}
typedef {{class_type}} {{class_type|flat|strip(' *')}};
{% if enum_gets or named_gets %}
{% for m in enum_gets %}
extern "C" {{m[2]}} {{'*' if 'Ptr' in m[2] else ''}} get_{{flatname}}_{{m[1]}}({{class_type}} * obj_);
{% endfor %}
{% for m in named_gets %}
extern "C" {{m[0]}} {{'*' if 'Ptr' in m[0] else ''}} get_{{flatname}}_{{m[1]}}({{class_type}} * obj_);
{% endfor %}
%SNIP%
{% for m in enum_gets %}
extern "C" {{m[2]}} {{'*' if 'Ptr' in m[2] else ''}} get_{{flatname}}_{{m[1]}}({{class_type}} * obj_)
{
    return RAW< {{m[2]}} >()((*obj_)->get({{m[0]}}::{{m[1]}}));
}
{% endfor %}
{% for m in named_gets %}
extern "C" {{m[0]}} {{'*' if 'Ptr' in m[0] else ''}} get_{{flatname}}_{{m[1]}}({{class_type}} * obj_)
{
    return RAW< {{m[0]}} >()((*obj_)->get{{m[1]}}());
}
{% endfor %}
%SNIP%
cdef extern from "daal4py_cpp.h":
    cdef cppclass {{class_type|flat|strip(' *')}}:
        pass
{% if not class_type.startswith('daal::'+ns) %}
{% endif %}
cdef extern from "daal4py_cpp.h":
{% for m in enum_gets %}
    cdef {{m[2]|d2cy}} get_{{flatname}}_{{m[1]}}({{class_type|flat}} obj_) except +
{% endfor %}
{% for m in named_gets %}
    cdef {{m[0]|d2cy}} get_{{flatname}}_{{m[1]}}({{class_type|flat}} obj_) except +
{% endfor %}

cdef class {{flatname}}:
    '''
    Properties:
    '''
    cdef {{class_type|flat}} c_ptr
    def __cinit__(self):
        pass
    def __dealloc__(self):
        del self.c_ptr
{% for m in enum_gets+named_gets %}
{% set rtype = m[2]|d2cy(False) if m in enum_gets else m[0]|d2cy(False) %}

    @property
    def {{m[1]}}(self):
{% if ('Ptr' in rtype and 'NumericTablePtr' not in rtype) or '__iface__' in rtype %}
{% set frtype=(rtype.strip(' *&')|flat(False)|strip(' *')).replace('Ptr', '')|lower %}
        ':type: {{frtype}}'
        res = {{frtype}}()
        res.c_ptr = get_{{flatname}}_{{m[1]}}(self.c_ptr)
        return res
{% else %}
        ':type: {{'Numpy array' if 'NumericTablePtr' in rtype else rtype}}'
        res = get_{{flatname}}_{{m[1]}}(self.c_ptr)
        return {{'<object>make_nda(res)' if 'NumericTablePtr' in rtype else 'res'}}
{% endif %}
{% endfor %}

hpat_spec.append({
    'pyclass': {{flatname}},
    'c_name' : '{{flatname}}',
    'attrs'  : [
{% for m in enum_gets+named_gets %}
                ('{{m[1]}}', '{{m[1]|d2hpat(m[2], flatname) if m in enum_gets else m[1]|d2hpat(m[0], flatname)}}'),
{% endfor %}
]})
{% else %}
%SNIP%
%SNIP%
{% endif %}
{% if not class_type.startswith('daal::'+ns) %}
cdef extern from "daal4py_cpp.h":
    ctypedef {{class_type|flat(False)}} {{(ns+'::'+splitname[-1])|flat|strip(' *')}}

{% set alias = (ns+'::'+splitname[-1]).replace('Ptr', '')|flat|lower|strip(' *') %}
{% set actual = flatname %}
{{alias}} = {{actual}}

hpat_spec.append({
    'c_name' : '{{alias}}',
    'alias' : '{{actual}}',
})
{% endif %}
"""

# macro generating C++ class for DAAL interface classes
# accepts interface name and C++ type
gen_cpp_iface_macro = """
{% macro gen_cpp_iface(iface_name, iface_type) %}
class {{iface_name}}__iface__ : public algo_manager__iface__
{
public:
    typedef {{iface_type}} daal_type;
    typedef {{iface_type}} {{iface_name}}Ptr_type;
    virtual {{iface_name}}Ptr_type get_ptr() = 0; //{return {{iface_name}}Ptr_type();}
};

typedef {{iface_name}}__iface__ c_{{iface_name}}__iface__;

static {{iface_type}} to_daal(c_{{iface_name}}__iface__ * t) {return t ? t->get_ptr() : {{iface_type}}();}
{% endmacro %}
"""

# macro generating cython class for DAAL interface classes
# accepts interface name and C++ type
gen_cython_iface_macro = """
{% macro gen_cython_iface(iface_name, iface_type) %}
cdef extern from "daal4py_cpp.h":
    cdef cppclass c_{{iface_name}}__iface__:
        pass

#    ctypedef c_{{iface_name}}__iface__ c_{{iface_type|flat|strip(' *')}};

{% set inl = iface_name|lower + '__iface__' %}
cdef class {{inl}}:
    cdef c_{{iface_name}}__iface__ * c_ptr

    def __cinit__(self):
        self.c_ptr = NULL

    def __dealloc__(self):
        del self.c_ptr


hpat_spec.append({
    'pyclass'     : {{inl}},
    'c_name'      : '{{inl}}',
})
{% endmacro %}
"""

# macro generating typedefs in manager classes, e.g. algo and result types
# note that we will have multiple result types in dist-mode: each step has its own
gen_typedefs_macro = """
{% macro gen_typedefs(ns, template_decl, template_args, mode="Batch", suffix="b", step_spec=None) %}
{% set disttarg = (step_spec.name.rsplit('__', 1)[0] + ', ') if step_spec.name else "" %}
{% if template_decl|length > 0  %}
    typedef {{ns}}::{{mode}}<{{disttarg + ', '.join(template_args)}}> algo{{suffix}}_type;
{% else %}
    typedef {{ns}}::{{mode}} algo{{suffix}}_type;
{% endif %}
{% if step_spec %}
    typedef {{step_spec.iomanager}}< algo{{suffix}}_type, {{', '.join(step_spec.input)}}, {{step_spec.output}}{{(","+",".join(step_spec.iomargs)) if step_spec.iomargs else ""}} > iom{{suffix}}_type;
{% else %}
{% if iombatch %}
    typedef {{iombatch}} iom{{suffix}}_type;
{% else %}
    typedef IOManager< algo{{suffix}}_type, services::SharedPtr< typename algo{{suffix}}_type::InputType >, services::SharedPtr< typename algo{{suffix}}_type::ResultType > > iom{{suffix}}_type;
{% endif %}
{% endif %}
{%- endmacro %}
"""

# macro for generate an algorithm instance with name algo$suffix
# This can also be used for steps of distributed mode
# set member=True if you want a init a member var (shared pointer)
gen_inst_algo = """
{% macro gen_inst(ns, params_req, params_opt, params_get, create, suffix="", step_spec=None, member=False) %}
{% set algo = 'algo' + suffix %}
{% if step_spec.construct %}
{% set ctor = create + '(' + step_spec.construct + ')' %}
{% elif params_req|length > 0  %}
{% set ctor = create + ('(to_daal(_' + '), to_daal(_'.join(params_req.values()) + '))') %}
{% else %}
{% set ctor = create + '()' %}
{% endif %}
{% if member %}
_algo{{suffix}}{{' = (' if create else '.reset(new '}}{{algo}}_type{{ctor}});
{% elif create %}
auto {{algo}} = {{algo}}_type{{ctor}};
{% else %}
auto {{algo}}_obj = {{algo}}_type{{ctor}};
        {{algo}}_type * {{algo}} = &{{algo}}_obj;
{% endif %}
{% if (step_spec == None or step_spec.params) and params_get and params_opt|length %}
        init_parameters({{('_' if member else '')+algo}}->{{params_get}});
{% else %}
        // skipping parameter initialization
{% endif %}
{%- endmacro %}
"""

# macro to generate the body of a compute function (batch and distributed)
gen_compute_macro = gen_inst_algo + """
{% macro gen_compute(ns, input_args, params_req, params_opt, suffix="", step_spec=None, tonative=True, iomtype=None) %}
{% set iom = iomtype if iomtype else "iom"+suffix+"_type" %}
{% if step_spec %}
{% if step_spec.addinput %}
(const std::vector< typename {{iom}}::input1_type > & input{{', ' + step_spec.extrainput if step_spec.extrainput else ''}})
    {
        {{gen_inst(ns, params_req, params_opt, params_get, create, suffix, step_spec)}}
        int nr = 0, i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            if(*data) {
                algo{{suffix}}->input.add({{step_spec.addinput}}, to_daal(*data));
                ++nr;
            }
        }
        if(nr == 0 ) return typename {{iom}}::result_type();
{% else %}
({% for ia in step_spec.input %}const typename {{iom}}::input{{loop.index}}_type & input{{loop.index}}{{'' if loop.last else ', '}}{% endfor %}{{', ' + step_spec.extrainput if step_spec.extrainput else ''}})
    {
        {{gen_inst(ns, params_req, params_opt, params_get, create, suffix, step_spec)}}
{% for ia in step_spec.input %}
        if(input{{loop.index}}) algo{{suffix}}->input.set({{step_spec.setinput[loop.index0]}}, to_daal(input{{loop.index}}));
{% endfor %}
{% endif %}
{% if step_spec.staticinput %}{% for ia in step_spec.staticinput %}
        if(! use_default(_{{ia[1]}})) algo{{suffix}}->input.set({{ia[0]}}, to_daal(_{{ia[1]}}));
{% endfor %}{% endif %}
{% else %}
(bool setup_only = false)
    {
        auto algo{{suffix}} = _algo{{suffix}};

{% for ia in input_args %}
{% if "TableOrFList" in ia[2] %}
        if(!_{{ia[1]}}->table && _{{ia[1]}}->file.size()) _{{ia[1]}}->table = readCSV(_{{ia[1]}}->file);
        if(_{{ia[1]}}->table) algo{{suffix}}->input.set({{ia[0]}}, _{{ia[1]}}->table);
{% else %}
        if(_{{ia[1]}}) algo{{suffix}}->input.set({{ia[0]}}, to_daal(_{{ia[1]}}));
{% endif %}
{% endfor %}

        if(setup_only) return typename iomb_type::result_type();
{% endif %}

        algo{{suffix}}->compute();
{% if step_spec %}
        if({{iom}}::needsFini()) {
            algo{{suffix}}->finalizeCompute();
        }
{% endif %}
{% if tonative %}
        typename iomb_type::result_type daalres({{iom}}::getResult(*algo{{suffix}});
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
{% else %}
        return {{iom}}::getResult(*algo{{suffix}});
{% endif %}
    }
{%- endmacro %}
"""

# generates the de-templetized *__iface__ struct with providing generic compute(...)
algo_iface_template = """
struct {{algo}}__iface__ : public {{iface[0] if iface[0] else 'algo_manager'}}__iface__
{
    bool _distributed;
    {{algo}}__iface__(bool d=false) : _distributed(d) {}
    virtual {{result_map.class_type}} * compute({{(',\n'+' '*(23+(result_map.class_type|length))).join(iargs_decl|cppdecl)}},
                                                bool setup_only = false) {assert(false);}
};
"""

# generates "manager" class for managing distributed and batch modes of a given algo
manager_wrapper_template = gen_typedefs_macro + gen_compute_macro + """
{% if base %}
// The type used in cython
typedef {{algo}}__iface__  c_{{algo}}_manager__iface__;

// The algo creation function
extern "C" {{algo}}__iface__ * mk_{{algo}}({{pargs_decl|cpp_decl(pargs_call, template_decl, 52+2*(algo|length))}});
{% endif %}

{% if template_decl  %}
template<{% for x in template_decl %}{{template_decl[x]['template_decl'] + ' ' + x + ('' if loop.last else ', ')}}{% endfor %}>
{% endif %}
struct {{algo}}_manager{% if template_decl and template_args and template_decl|length != template_args|length %}<{{', '.join(template_args)}}>{% endif %} : public {{algo}}__iface__
{% if template_args %}
{
{{gen_typedefs(ns, template_decl, template_args, mode="Batch")}}
{% for i in pargs_decl %}    {{' _'.join((i|cppdecl(True)).replace('&', '').strip().replace('__iface__ *', '__iface__::daal_type').rsplit(' ', 1))}};
{% endfor %}
{% for i in iargs_decl %}    {{' _'.join((i|cppdecl(True)).replace('&', '').strip().replace('__iface__ *', '__iface__::daal_type').rsplit(' ', 1))}};
{% endfor %}
    daal::services::SharedPtr< algob_type > _algob;

    {{algo}}_manager({{(',\n'+' '*(13+algo|length)).join((pargs_decl + ['bool distributed = false'])|cppdecl)}})
        : {{algo}}__iface__(distributed)
{% for i in pargs_call %}
        , _{{i}}({{'to_daal('+i+')' if '__iface__' in (pargs_decl[loop.index0]|cppdecl(True)) else i}})
{% endfor %}
        , _algob()
    {
        {{gen_inst(ns, params_req, params_opt, params_get, create, suffix="b", member=True)}}
    }

#ifdef _DIST_
    {{algo}}_manager() :
        {{algo}}__iface__(true)
{% for i in args_call %}
        , _{{i}}()
{% endfor %}
        , _algob()
    {
        {{gen_inst(ns, params_req, params_opt, params_get, create, suffix="b", member=True)}}
    }
#endif

private:
{% if params_opt|length %}
    template< typename PType >
    void init_parameters(PType & parameter)
    {
{% for p in params_opt %}
        if(! use_default(_{{p}})) parameter.{{p}} = to_daal({{params_opt[p].replace(p, '_'+p)}});
{% endfor %}
    }
{% endif %}

{% for ifc in iface if ifc %}
    virtual {{ifc}}__iface__::{{ifc}}Ptr_type get_ptr()
    {
        return _algob;
    }
{% endfor %}

    typename iomb_type::result_type batch{{gen_compute(ns, input_args, params_req, params_opt, suffix="b", iomtype=iombatch, tonative=False)}}

{% if step_specs %}
#ifdef _DIST_
    // Distributed computing
public:
{% for i in range(step_specs|length) %}
{{gen_typedefs(ns, template_decl, template_args, mode="Distributed", suffix=step_specs[i].name, step_spec=step_specs[i])}}
{% endfor %}

{% for i in range(step_specs|length) %}
{% set sname = "run_"+step_specs[i].name %}
    typename iom{{step_specs[i].name}}_type::result_type {{sname + gen_compute(ns, input_args, params_req, params_opt, suffix=step_specs[i].name, step_spec=step_specs[i], tonative=False)}}

{% endfor %}

    static const int NI = {{step_specs[0].inputnames|length}};

private:
    typename iomb_type::result_type distributed() // iom{{step_specs[-1].name}}_type::result_type
    {
        return {{pattern}}::{{pattern}}< {{algo}}_manager< {{', '.join(template_args)}} > >::compute(to_daal(_{{'), to_daal(_'.join(step_specs[0].inputnames)}}), *this);
    }
#endif
{% endif %}

public:
#ifdef _DIST_
    virtual void serialize(CnC::serializer & ser)
    {
        ser
{% for i in pargs_call %}            & _{{i.rsplit('=', 1)[0]}}
{% endfor %};
    }
#endif

public:
    typename iomb_type::result_type * compute({{(',\n'+' '*46).join(iargs_decl|cppdecl)}},
                                              bool setup_only = false)
    {
{% for i in iargs_call %}        _{{i}} = {{i}};
{% endfor %}

#ifdef _DIST_
        typename iomb_type::result_type daalres = {{'_distributed ? distributed() : batch(setup_only);' if dist else 'batch(setup_only);'}}
        return new typename iomb_type::result_type(daalres);
#else
        return new typename iomb_type::result_type(batch(setup_only));
#endif
    }
};
{% if step_specs %}
#ifdef _DIST_
namespace CnC {
{% if template_decl|length > 0  %}
template<{% for x in template_decl %}{{template_decl[x]['template_decl'] + ' ' + x + ('' if loop.last else ', ')}}{% endfor %}>
{% endif %}
    static inline void serialize(serializer & ser, {{algo}}_manager{% if template_args|length %}<{{', '.join(template_args)}}>{% endif %} *& t)
    {
        int sz = ser.is_packing() && t == NULL ? 0 : 1;
        ser & sz;
        if(sz) ser & chunk< {{algo}}_manager{% if template_args|length %}<{{', '.join(template_args)}}>{% endif %} >(t, sz);
        else if(ser.is_unpacking()) t = NULL;
    }
}
#endif
{% endif %}
{% else %}
{};
{% endif %}
"""

# generates cython class wrappers for given algo
# also generates defs for __iface__ class
parent_wrapper_template = """
cdef extern from "daal4py.h":
    # declare the C++ equivalent of the manager__iface__ class, providing de-templatized access to compute
    cdef cppclass c_{{algo}}_manager__iface__{{'(c_'+iface[0]+'__iface__)' if iface[0] else ''}}:
        {{result_map.class_type|flat}} compute({{(',\n'+' '*(29+algo|length)).join(iargs_decl|d2ext)}},
        {{' '*(21+algo|length)}}const bool setup_only) except +


cdef extern from "daal4py_cpp.h":
    # declare the C++ construction function. Returns the manager__iface__ for access to de-templatized constructor
    cdef c_{{algo}}_manager__iface__ * mk_{{algo}}({{pargs_decl|cy_ext_decl(pargs_call, template_decl, 45+2*(algo|length))}}) except +


# this is our actual algorithm class for Python
cdef class {{algo}}{{'('+iface[0]|lower+'__iface__)' if iface[0] else ''}}:
    '''
    {{algo}}
    {{pargs_decl|cy_decl(pargs_call, template_decl, 18)|sphinx}}
    '''
    # Init simply forwards to the C++ construction function
    def __cinit__(self,
                  {{pargs_decl|cy_decl(pargs_call, template_decl, 18)}}):
        self.c_ptr = mk_{{algo}}({{pargs_decl|cy_call(pargs_call, template_decl, 45+(algo|length))}})

{% if not iface[0] %}
    # the C++ manager__iface__ (de-templatized)
    cdef c_{{algo}}_manager__iface__ * c_ptr

    def __dealloc__(self):
        del self.c_ptr
{% endif %}
    # compute simply forwards to the C++ de-templatized manager__iface__::compute
    def compute(self,
{% for a in iargs_call %}
{{' '*16 + a|cydecl(args_decl[loop.index0]) + ('):' if loop.last else ',')}}
{% endfor %}
        algo = <c_{{algo}}_manager__iface__ *>self.c_ptr
        # we cannot have a constructor accepting a c-pointer, so we split into construction and setting pointer
        res = {{result_map.class_type.replace('Ptr', '')|d2cy(False)|lower}}()
        res.c_ptr = deref(algo).compute(
{%- for a in iargs_call -%}
{{('' if loop.first else ' '*40) + a|cycall(iargs_decl[loop.index0]) + (', False)' if loop.last else ',')}}
{% endfor %}
        return res

{% if add_setup %}
    # setup forwards to the C++ de-templatized manager__iface__::compute(..., setup_only=true)
    def setup(self,
{% for a in iargs_call %}
{{' '*14 + a|cydecl(args_decl[loop.index0]) + ('):' if loop.last else ',')}}
{% endfor %}
        algo = <c_{{algo}}_manager__iface__ *>self.c_ptr
        deref(algo).compute(
{%- for a in iargs_call -%}
{{('' if loop.first else ' '*28) + a|cycall(iargs_decl[loop.index0]) + (', True)' if loop.last else ',')}}
{% endfor %}
        return None
{% endif %}
"""

# generates the C++ algorithm construction function
# it all it does is dispatching to the template managers from given arguments
# tfactory is a recursive jinja2 macro to handle any number of template args
algo_wrapper_template = """
{% macro tfactory(tmpl_spec, prefix, pcallargs, dist=False, args=[], indent=4) %}
{{" "*indent}}if( false ) {;}
{% for a in tmpl_spec[0][1]['values'] %}
{% if tmpl_spec[0][1]['values']|length > 1 %}
{{" "*indent}}else if({{tmpl_spec[0][0]}} == "{{a.rsplit('::',1)[-1]}}") {
{% else %}
{{" "*indent}}else {
{% endif %}
{% if tmpl_spec|length == 1 %}
{% set algo_type = prefix + '<' + ', '.join(args+[a]) + ' >' %}
{{" "*(indent+4)}}return new {{algo_type}}({{', '.join(pcallargs + ['distributed'])}});
{{" "*(indent)}}}
{% else %}
{{tfactory(tmpl_spec[1:], prefix, pcallargs, dist, args+[a], indent+4)}}
{{" "*(indent)}}}
{% endif %}
{% endfor %}
{%- endmacro %}

extern "C" {{algo}}__iface__ * mk_{{algo}}({{pargs_decl|cpp_decl(pargs_call, template_decl, 56+2*(algo|length))}})
{
{% if template_decl %}
{{tfactory(template_decl.items()|list, algo+'_manager', pargs_call, dist=dist)}}
    throw std::invalid_argument("no equivalent(s) for C++ template argument(s) in mk_{{algo}}");
{% else %}
    return new {{algo}}_manager({{', '.join(pargs_call)}}, distributed);
{% endif %}
}

extern "C" void * compute_{{algo}}({{(',\n'+' '*(27+algo|length)).join([algo+'__iface__ * algo']+iargs_decl|hpatdecl)}})
{
#ifdef _DIST_
    algo->_distributed = c_num_procs() > 0;
#endif
    void * res = algo->compute(
{% for a in iargs_decl %}
{% set comma = ');' if loop.last else ',' %}
{% set an = a.rsplit('=', 1)[0].strip().rsplit(' ', 1)[-1] %}
{% if "TableOrFList" in a %}
{{' '*34}}new TableOrFList(daal::data_management::HomogenNumericTable< double >::create({{an}}_p, {{an}}_d2, {{an}}_d1)){{comma}}
{% else %}
{{' '*34 + an}}{{comma}}
{% endif %}
{% endfor %}
    return res;
};
"""

# We need to register all possible context types to CnC
# As the algos are templetized we need to do this for all template-instantiations of algo managers.
# For ifaces we need to construct derived classes from iface pointers.
# For this we use CnC factory as well, so need register the algo_managers, too.
algo_types_template = """
{% macro tfactory(tmpl_spec, prefix, args=[]) %}
{% for a in tmpl_spec[0][1]['values'] %}
{% if tmpl_spec|length == 1 %}
CnC::Internal::factory::subscribe< typename {{pattern}}::{{pattern}}< {{prefix + '<' + ', '.join(args+[a]) + ' > >::context_type'}} >();
{% else %}
{{tfactory(tmpl_spec[1:], prefix, args+[a])}}
{% endif %}
{% endfor %}
{%- endmacro %}

{% if step_specs %}
{% if template_decl %}
{{tfactory(template_decl.items()|list, algo+'_manager')}}
{% else %}
CnC::Internal::factory::subscribe< {{pattern}}::{{pattern}}< {{algo}}_manager >::context_type >();
{% endif %}
{% endif %}
"""

# Create initialization code.
# Requries {{subscriptions}}
init_template = '''
typedef daal::data_management::interface1::NumericTablePtr NumericTablePtr;

#ifdef _DIST_
typedef CnC::Internal::dist_init init_type;
init_type * initer = NULL;

struct fini
{
    ~fini()
    {
        if(initer) delete initer;
        initer = NULL;
    }
};
static fini _fini;

extern "C" {

void c_daalinit(bool spmd, int flag)
{
    if(initer) delete initer;
    auto subscriber = [](){
{{subscriptions.rstrip()}}
    };
    initer = new init_type(subscriber, flag, spmd);
}

void c_daalfini()
{
    if(initer) delete initer;
    initer = NULL;
}

size_t c_num_procs()
{
    return CnC::tuner_base::numProcs();
}

size_t c_my_procid()
{
    return CnC::tuner_base::myPid();
}

} // extern "C"
#endif //_DIST_

'''

# generate a D4PSpec
hpat_spec_template = '''
hpat_spec.append({
    'pyclass'     : {{algo}},
    'c_name'      : '{{algo}}',
    'params'      : [{{pargs_decl|hpat_spec(pargs_call, template_decl, 21)}}],
    'input_types' : {{iargs_decl|hpat_input_spec(step_specs[0].inputdists if step_specs else None)}},
    'result_dist' : {{"'REP'" if step_specs else "'OneD'"}}
})
'''

##################################################################################
# A set of jinja2 filters to convert arguments, types etc which where extracted
# from DAAL C++ headers to cython syntax and/or C++ for our own code
##################################################################################
def flat(t, cpp=True):
    '''Flatten C++ name, leaving only what's needed to disambiguate names.
       E.g. stripping of leading namespaces and replaceing :: with _
    '''
    def _flat(ty):
        def __flat(typ):
            nn = typ.split('::')
            if nn[0] == 'daal':
                if nn[1] == 'algorithms':
                    r = '_'.join(nn[2:])
                else:
                    r = '_'.join(nn[1:])
            elif nn[0] == 'algorithms':
                r = '_'.join(nn[1:])
            else:
                r = '_'.join(nn)
            return ('c_' if cpp and typ.endswith('__iface__') else '') + r + (' *' if cpp and any(typ.endswith(x) for x in ['__iface__', 'Ptr']) else '')
        ty = ty.replace('daal::algorithms::kernel_function::KernelIfacePtr', 'services::SharedPtr<kernel_function::KernelIface>')
        ty = re.sub(r'(daal::)?(algorithms::)?(engines::)?EnginePtr', 'services::SharedPtr<engines::BatchBase>', ty)
        ty = re.sub(r'(daal::)?(algorithms::)?(sum_of_functions::)?BatchPtr', 'services::SharedPtr<sum_of_functions::Batch>', ty)
        ty = re.sub(r'(daal::)?services::SharedPtr<([^>]+)>', r'\2__iface__', ty)
        return ' '.join([__flat(x).replace('const', '') for x in ty.split(' ')])
    return [_flat(x) for x in t] if isinstance(t,list) else _flat(t)

def d2cy(ty, cpp=True):
    def flt(t, cpp):
        return flat(t, cpp).replace('lambda', 'lambda_')
    return [flt(x,cpp) for x in ty] if isinstance(ty,list) else flt(ty,cpp)


def d2ext(ty, cpp=True):
    def flt(t):
        return flat(t, cpp).split('=')[0].strip().replace('lambda', 'lambda_')
    return [flt(x) for x in ty] if isinstance(ty,list) else flt(ty)

def gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt):
    '''Generate list of arguments/paramters from algorithm arguments/paramters.
       Brings required and optional arguments in the right order and
       applies a filter
    '''
    r = ''
    for a in range(len(pargs_decl)):
        if '=' not in pargs_decl[a]:
            r += ' '*indent + flt(pargs_call[a], pargs_decl[a]) + ',\n'
    for ta in template_decl:
        if not template_decl[ta]['default']:
            r += ' '*indent + flt(ta, 'const std::string & ' + ta) + ',\n'
    for ta in template_decl:
        if template_decl[ta]['default']:
            r += ' '*indent + flt(ta, 'const std::string & ' + ta + ' = "' + template_decl[ta]['default'].rsplit('::',1)[-1] + '"') + ',\n'
    for a in range(len(pargs_decl)):
        if '=' in pargs_decl[a]:
            r += ' '*indent + flt(pargs_call[a], pargs_decl[a]) + ',\n'
    return r.lstrip() + ' '*indent + flt('distributed', 'bool distributed = False')

def cy_ext_decl(pargs_decl, pargs_call, template_decl, indent):
    def flt(arg, typ):
        return d2cy(typ).rsplit('=', 1)[0].strip()
    return gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt)

def cydecl(arg, typ):
    if 'TableOrFList' in typ:
        return arg
    if 'NumericTablePtr' in typ:
        return arg + (' = None' if '=' in typ else '')
    r = d2cy(typ, False).replace('Ptr ', ' ').strip().rsplit('=', 1)
    if 'std::string' in typ:
        return arg + ' = ' + r[-1]
    tv = r[0].strip().rsplit(' ', 1)
    if len(tv):
        tv = " ".join([tv[0].lower()] + tv[1:])
    return tv if len(r)==1 else '='.join([tv, r[1]])

def cy_decl(pargs_decl, pargs_call, template_decl, indent):
    def flt(arg, typ):
        return cydecl(arg, typ)
    return gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt)

def cycall(arg, typ):
    if 'TableOrFList' in typ:
        return 'new TableOrFList(<PyObject *>' + arg + ')'
    if 'NumericTablePtr' in typ:
        return 'make_nt(<PyObject *>' + arg + ')'
    if 'Ptr' in typ:
        return arg + '.c_ptr if ' + arg + ' != None else <' + flat(typ.rsplit('=', 1)[0].strip().rsplit(' ', 1)[0]) + '>0'
    if 'std::string' in typ:
        return 'to_std_string(<PyObject *>' + arg + ')'
    return arg.replace('lambda', 'lambda_')

def cy_call(pargs_decl, pargs_call, template_decl, indent):
    def flt(arg, typ):
        return cycall(arg, typ)
    return gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt)

def cppdecl(ty, noconst=False):
    def flt(typ):
        if 'Ptr' in typ:
            typ = flat(typ, True)
        if noconst or 'std::string' not in typ:
            typ = typ.replace('const', '')
        return typ.split('=')[0].strip()
    return [flt(x) for x in ty] if isinstance(ty, list) else flt(ty)

def cpp_decl(pargs_decl, pargs_call, template_decl, indent):
    def flt(arg, typ):
        return cppdecl(typ)
    return gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt)

def d2hpat(arg, ty, fn):
    def flt(arg, t):
        rtype = d2cy(t)
        if fn in hpat_types and arg in hpat_types[fn]:
            return hpat_types[fn][arg]
        return 'dtable_type' if 'NumericTablePtr' in rtype else rtype.replace('ModelPtr', 'model').replace(' ', '')
    return [flt(x,y) for x,y in zip(arg, ty)] if isinstance(ty,list) else flt(arg, ty)

def hpatdecl(ty):
    def flt(typ):
        if "TableOrFList" in typ:
            an = typ.rsplit('=', 1)[0].strip().rsplit(' ', 1)[-1]
            return 'double * ' + an + '_p, size_t ' + an + '_d1, size_t ' + an + '_d2'
        if 'Ptr' in typ:
            typ = flat(typ)
        return typ.split('=')[0].replace('const', '').strip()
    return [flt(x) for x in ty] if isinstance(ty, list) else flt(ty)

def hpat_input_spec(ty, dists):
    def flt(typ, dist):
        an = typ.rsplit('=', 1)[0].strip().rsplit(' ', 1)[-1]
        if "TableOrFList" in typ:
            return (an, 'dtable_type', dist)
        if 'Ptr' in typ:
            typ = flat(typ, False)
        return (an, typ.replace('const', '').strip().split()[0].replace('ModelPtr', 'model'), dist)

    if dists == None:
        dists = ['REP' if 'model' in x else 'OneD' for x in ty]
    assert len(dists) == len(ty)
    return [flt(x,y) for x,y in zip(ty, dists)]

def hpat_spec(pargs_decl, pargs_call, template_decl, indent):
    def flt(arg, typ):
        if 'Ptr' in typ:
            typ = flat(typ, False)
        st = typ.split('=')
        ref = '&' if '&' in st[0] else ('*' if '*' in st[0] else '')
        ret = "'{}', '{}'".format(arg.replace('lambda', 'lambda_'), st[0].replace('const', '').strip().split()[0].lower()+ref)
        return '({}, {})'.format(ret, st[1]) if len(st) > 1 else '({})'.format(ret)
    return gen_algo_args(pargs_decl, pargs_call, template_decl, indent, flt)

def sphinx(st):
    def flt(s):
        lst = s.rsplit('=', 1)
        rstr = '[, ' if len(lst) > 1 else ', '
        oval = ' = ' + lst[1].strip() + ']' if len(lst) > 1 else ''
        dflt = ' [optional, default: '+lst[1].strip()+']' if len(lst) > 1 else ''
        llst = lst[0].split()
        if len(llst) == 1:
            return (rstr + lst[0].strip() +': str' + oval, '   :param str ' + lst[0].strip() + ':' + dflt)
        elif len(llst) == 2:
            return (rstr + llst[1].strip() + ': ' + llst[0].strip() + oval, '   :param ' + llst[0].strip() + ' ' + llst[1].strip() + ':' + dflt)
        else:
            assert False, 'oops' + s

    all = [flt(x) for x in st.split(',')]
#    return '(' + ''.join([x[0] for x in all]).strip(' ,') + ')\n\n       ' + '\n'.join([x[1] for x in all]).strip(' ,')
    return '\n   ' + '\n'.join([x[1] for x in all]).strip(' ,')

jenv = jinja2.Environment(trim_blocks=True)
jenv.filters['match'] = lambda a, x : [x for x in a if s in x]
jenv.filters['d2cy'] = d2cy
jenv.filters['d2ext'] = d2ext
jenv.filters['flat'] = flat
jenv.filters['cydecl'] = cydecl
jenv.filters['cycall'] = cycall
jenv.filters['cy_ext_decl'] = cy_ext_decl
jenv.filters['cy_decl'] = cy_decl
jenv.filters['cy_call'] = cy_call
jenv.filters['cppdecl'] = cppdecl
jenv.filters['cpp_decl'] = cpp_decl
jenv.filters['d2hpat'] = d2hpat
jenv.filters['hpatdecl'] = hpatdecl
jenv.filters['hpat_spec'] = hpat_spec
jenv.filters['hpat_input_spec'] = hpat_input_spec
jenv.filters['strip'] = lambda s, c : s.strip(c)
jenv.filters['sphinx'] = sphinx

class wrapper_gen(object):
    def __init__(self, ac, ifaces):
        self.algocfg = ac
        self.ifaces = ifaces

    def gen_headers(self):
        """
        return code for initing
        """
        cpp = "#ifndef DAAL4PY_CPP_INC_\n#define DAAL4PY_CPP_INC_\n#include <daal4py_dist.h>\n\ntypedef daal::data_management::interface1::NumericTablePtr NumericTablePtr;"
        pyx = ''
        for i in self.ifaces:
            tstr = gen_cython_iface_macro + '{{gen_cython_iface("' + i + '", "' + self.ifaces[i] + '")}}\n'
            t = jenv.from_string(tstr)
            pyx += t.render({}) + '\n'
            tstr = gen_cpp_iface_macro + '{{gen_cpp_iface("' + i + '", "' + self.ifaces[i] + '")}}\n'
            t = jenv.from_string(tstr)
            cpp += t.render({}) + '\n'

        return (cpp, cython_header + pyx)

    def gen_hlargs(self, template_decl, args_decl):
        """
        Generates a list of tuples, one for each HLAPI argument: (name, type, default)
        """
        res = []
        for a in args_decl:
            if '=' not in a:
                tmp = a.strip().rsplit(' ', 1)
                res.append((tmp[1], tmp[0], None))
        for ta in template_decl:
            if not template_decl[ta]['default']:
                res.append((ta, 'string', None))
        for ta in template_decl:
            if template_decl[ta]['default']:
                res.append((ta, 'string', template_decl[ta]['default']))
        for a in args_decl:
            if '=' in a:
                tmp1 = a.strip().rsplit('=', 1)
                tmp2 = tmp1[0].strip().rsplit(' ', 1)
                res.append((tmp2[1], tmp2[0], tmp1[1]))
        return res


    ##################################################################################
    def gen_modelmaps(self, ns, algo):
        """
        return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        """
        jparams = self.algocfg[ns + '::' + algo]['model_typemap']
        if len(jparams) > 0:
            jparams['ns'] = ns
            jparams['algo'] = algo
            t = jenv.from_string(typemap_wrapper_template)
            return (t.render(**jparams) + '\n').split('%SNIP%')
        return '', '', ''


    ##################################################################################
    def gen_resultmaps(self, ns, algo):
        """
        Generates typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
          -
        Looks up Return type and then target-language independently creates lists of its content.
        """
        jparams = self.algocfg[ns + '::' + algo]['result_typemap']
        if len(jparams) > 0:
            jparams['ns'] = ns
            jparams['algo'] = algo
            t = jenv.from_string(typemap_wrapper_template)
            return (t.render(**jparams) + '\n').split('%SNIP%')
        return '', '', ''

    ##################################################################################
    def gen_wrapper(self, ns, algo):
        """
        Here we actually generate the wrapper code. Separating this from preparation
        allows us to cross-reference between algos, for example for multi-phased algos.

        We combine the argument (template, input, parameter) information appropriately.
        We take care of the right order and bring them in the right format for our jinja templates.
        We pass them to the templates in a dict jparams, used a globals vars for jinja.

        Handling single-phased algos only which are not part of a multi-phased algo
        """
        cfg = self.algocfg[ns + '::' + algo]
        cpp_begin, pyx_begin, pyx_end, typesstr = '', '', '', ''
        hlargs = []

        cpp_map, cpp_end, pyx_map = self.gen_modelmaps(ns, algo)
        a, b, c = self.gen_resultmaps(ns, algo)
        cpp_map += a
        cpp_end += b
        pyx_map += c

        if len(cfg['params']) == 0:
            return (cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end, typesstr, hlargs)


        jparams = cfg['params'].copy()
        jparams['create'] = cfg['create']
        jparams['add_setup'] = cfg['add_setup']
        jparams['model_maps'] = cfg['model_typemap']
        jparams['result_map'] = cfg['result_typemap']
        jparams['pargs_decl'] = jparams['decl_req'] + jparams['decl_opt']
        jparams['args_decl']  = jparams['iargs_decl'] + jparams['pargs_decl']
        jparams['pargs_call'] = jparams['call_req'] + jparams['call_opt']
        jparams['args_call']  = jparams['iargs_call'] + jparams['pargs_call']
        tdecl = cfg['sparams']

        t = jenv.from_string(algo_iface_template)
        cpp_begin += t.render(**jparams) + '\n'
        # render all specializations
        i = 0
        for td in tdecl:
            # Last but not least, we need to provide the template parameter specs
            jparams['template_decl'] = td['template_decl']
            jparams['template_args'] = td['template_args']
            jparams['params_req'] = td['params_req']
            jparams['params_opt'] = td['params_opt']
            jparams['params_get'] = td['params_get']
            # Very simple for specializations
            # but how do we pass only the required args to them from the wrapper?
            # we could have the full input list, but that doesn't work for required parameters
            jparams['base'] = (i == 0)
            assert td['template_args'] != None or jparams['base'], "eerr"
            if 'dist' in cfg:
                # a wrapper for distributed mode
                assert len(tdecl) == 1
                jparams.update(cfg['dist'])
                jparams['dist'] = True
            t = jenv.from_string(manager_wrapper_template)
            cpp_begin += t.render(**jparams) + '\n'
            if td['pargs'] == None:
                t = jenv.from_string(hpat_spec_template)
                pyx_begin += t.render(**jparams) + '\n'
                # this is our actual API wrapper, only once per template (covering all its specializations)
                # the parent class
                t = jenv.from_string(parent_wrapper_template)
                pyx_end += t.render(**jparams) + '\n'
                # the C function generating specialized classes
                t = jenv.from_string(algo_wrapper_template)
                cpp_end += t.render(**jparams) + '\n'
                hlargs += self.gen_hlargs(jparams['template_decl'], jparams['args_decl'])

            t = jenv.from_string(algo_types_template)
            typesstr += t.render(**jparams) + '\n'
            i = i+1

        return (cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end, typesstr, hlargs)


    ##################################################################################
    def gen_footers(self, subscriptions):
        """ generate code for initing  CnC.
            Requires list of CnC context subscriptions."""
        jparams = {'subscriptions' : subscriptions}
        if len(jparams) > 0:
            t = jenv.from_string(init_template)
            cpp = re.sub(r'[\n\s]+CnC::I', '\n        CnC::I', t.render(**jparams)) + '\n'
        return (cpp, '')
