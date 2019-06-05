#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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

# HPAT support for daal4py - an easy-to-use ML API (to Intel(R) DAAL).
#
# We provide a factory which creates all numba/HPAT code needed to compile/distribute daal4py code.
# Given a algorithm specification (see list at end of the file) it generates numba types
# and lowering stuff for function calls (construction and compute) and member accesses
# (to attributes of Result/Model).
#
# Algorithm/Result/Model objects simply get lowered to opaque pointers.
# Attribute access gets redirected to DAAL's actual accessor methods.
#
# TODO:
#   - sub-classing: parameters of '__interface__' types must accept derived types
#   - boxing/unboxing of algorithms
#   - GC: result/model objects returned by daal4py wrappers are newly allocated shared pointers, need to get gc'ed
#   - see fixme's below

import numpy as np
from numpy import nan
from numba import types, cgutils, objmode, njit
from numba.extending import (intrinsic, typeof_impl, overload, overload_method,
                             overload_attribute, box, unbox, models, register_model,
                             NativeValue, get_cython_function_address)
from numba.targets.imputils import impl_ret_new_ref
from numba.typing.templates import signature
from collections import namedtuple
import warnings
from numba.types import unicode_type
from hpat.distributed_analysis import DistributedAnalysis, Distribution as DType
from llvmlite import ir as lir
from numba.targets.arrayobj import _empty_nd_impl
from hpat.str_ext import gen_unicode_to_std_str, unicode_to_std_str, std_str_type, del_str
from hpat.hiframes.pd_dataframe_ext import DataFrameType
import ctypes
import hpat


##############################################################################
##############################################################################
import daal4py
from daal4py import NAN32, NAN64, hpat_spec
import llvmlite.binding as ll

def open_daal4py():
    '''open daal4py library and load C-symbols'''
    import os
    import glob

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(daal4py.__file__))), '_daal4py.c*')
    lib = glob.glob(path)
    assert len(lib) == 1, "Couldn't find/determine daal4py library ({} -> {})".format(path, lib)

    # just load the whole thing
    ll.load_library_permanently(lib[0])


##############################################################################
##############################################################################

# short-cut for Array type. TODO: we currently only support 2d-double arrays
dtable_type = types.Array(types.float64, 2, 'C')
ftable_type = types.Array(types.float32, 2, 'C')
itable_type = types.Array(types.intc, 2, 'C')

nptypes = {
    1: dtable_type,
    2: ftable_type,
    3: itable_type,
}
d4ptypes = {
    'dtable_type': 1,
    'ftable_type': 2,
    'itable_type': 3,
}

d4p_dtypes = {
    'REP'     : DType.REP,
    'Thread'  : DType.Thread,
    'TwoD'    : DType.TwoD,
    'OneD_Var': DType.OneD_Var,
    'OneD'    : DType.OneD,
    None      : None
}

@intrinsic
def nt2nd(typingctx, obj, dtype):
    assert(isinstance(dtype, types.scalars.IntegerLiteral))
    np_ary_type = nptypes[dtype.literal_value]
    def codegen(context, builder, sig, args):  # def nt2nd(context, builder, ptr, ary_type):
        '''Generate ir code to convert a pointer-to-daal-numeric-table to a ndarray'''
        # we need to prepare the shape array and a pointer
        shape_type = lir.ArrayType(lir.IntType(64), 2)
        shape = cgutils.alloca_once(builder, shape_type)
        data = cgutils.alloca_once(builder, lir.DoubleType().as_pointer())
        # we can now declare and call our conversion function
        fnty = lir.FunctionType(lir.VoidType(),
                                [lir.IntType(8).as_pointer(), # actually pointer to numeric table
                                 lir.DoubleType().as_pointer().as_pointer(),
                                 shape_type.as_pointer(),
                                 lir.IntType(8)])
        fn = builder.module.get_or_insert_function(fnty, name='to_c_array')
        builder.call(fn, [args[0], data, shape, context.get_constant(types.byte, dtype.literal_value)])
        # convert to ndarray
        shape = cgutils.unpack_tuple(builder, builder.load(shape))
        ary = _empty_nd_impl(context, builder, np_ary_type, shape)
        cgutils.raw_memcpy(builder, ary.data, builder.load(data), ary.nitems, ary.itemsize, align=1)
        # we are done!
        return impl_ret_new_ref(context, builder, np_ary_type, ary._getvalue())
    return np_ary_type(types.voidptr, types.intc), codegen

def _arr2C(inp):
    if isinstance(inp, types.Array) and inp.ndim == 2:
        descr = None
        if inp.dtype == types.float64:
            descr = int(float(1))
        elif inp.dtype == types.float32:
            descr = int(float(2))
        elif inp.dtype == types.intc:
            descr = int(float(3))
        if descr:
            def inp2d4p_array(inp):
                return (inp.ctypes.data, inp.shape[0], inp.shape[1], descr)
            return inp2d4p_array
    if isinstance(inp, types.misc.NoneType):
        def inp2d4p_none(inp):
            return (int(float(0)), int(float(0)), int(float(0)), int(float(1)))
        return inp2d4p_none
    return None


def arr2C(a):
    raise ValueError('Dummy function, only its @overload should be used')

@overload(arr2C)
def ovl_arr2C(inp):
    '''
    Convert a numpy array or None to daal4py's C array signature:
    (void* ptr, int64 ncols, int64 nrows, int64 layout)
    '''
    r = _arr2C(inp)
    if r != None:
        return r
    raise ValueError("Input type '{}' must be an array".format(inp))


@hpat.jit
def _get_vals(df):
    return df.values

def inp2d4p(a):
    raise ValueError('Dummy function, only its @overload should be used')

@overload(inp2d4p)
def ovl_inp2d4p(inp):
    '''
    Input arguments can be of different python types: numpy arrays, pandas DFs and files.
    For each input type this @overload generates distinct function, each returning a tuple of the same kind:
    (void* ptr, int64 ncols, int64 nrows, int64 layout)
    The tuple is passed to C/C++ where it gets decoded when generating the internal data_or_file struct.
    '''
    r = _arr2C(inp)
    if r != None:
        return r
    if isinstance(inp, DataFrameType):
        # TODO: avoid using 'values', instead create a list of arrays
        #       will become easier with better support for i/loc
        #       requires some changes in daal4py as we'll not pass a python object
        def inp2d4p_df(inp):
            ary = _get_vals(inp)
            return arr2C(ary)
        return inp2d4p_df
    if isinstance(inp, types.UnicodeType):
        def inp2d4p_str(inp):
            return (inp._data, int(float(0)), inp._length, int(float(0)))
        return inp2d4p_str
    raise ValueError("Input type '{}' not supported".format(inp))

# C function to convert a numpy array into a DAAL NT (pointer to shared ptr)
#nt_from_c_array = types.ExternalFunction('from_c_array', types.voidptr(types.voidptr, types.int64, types.int64, types.int64))

##############################################################################
##############################################################################
# Class configs.
# A specification defines a daal4py class, which can be an algorithm, a model or a result.
# The following information is needed:
#    - spec.pyclass is expected to be the actual daal4py class.
#    - spec.c_name provides the name of the class name as used in C.
#    - spec.params list of tuples (name, numba type, default) for the algo parameters (constructor) [algo only]
#    - spec.input_types: list of pairs for input arguments to compute: (numba type, distribution type) [algo only]
#    - spec.attrs: list of tuple (name, numba-type) representing result or model attributes [model/result only]
#    - spec.result_dist: distribution type of result [algo only]
#    - spec.has_setup: True if algo has setup method, False otherwise
# Note: input/attribute types can be actual numba types or strings.
#       In the latter case, the type is looked up in the list of 'own' factory-created types
#       At this point this requires that we can put the list in a canonical order...

D4PSpec = namedtuple('D4PSpec',
                     'pyclass c_name params input_types result_dist attrs has_setup')
# default values, only name is required
D4PSpec.__new__.__defaults__ = (None, None, None, DType.REP, None, False)

##############################################################################
##############################################################################
class algo_factory(object):
    '''
    This factory class accepts a configuration for a daal4py class.
    Providing all the numba/lowering stuff needed to compile the given algo:
      - algo construction
      - algo computation
      - attribute access (results and models)
    TODO: GC for shared pointers of result/model objects
    '''

    # list of types, so that we can reference them when dealing others
    all_nbtypes = {
        'list_numerictable' : dtable_type, # TODO: is in fact a list of tables!
        'dict_numerictable' : dtable_type, # TODO: is in fact a dict of tables!
        'data_or_file': dtable_type,       # TODO: table can have different types, input can be file
        'table'       : dtable_type,       # TODO: table can have different types
        'dtable_type' : dtable_type,
        'ftable_type' : ftable_type,
        'itable_type' : itable_type,
        'size_t'      : types.uint64,
        'int'         : types.int32,
        'double'      : types.float64,
        'float'       : types.float32,
        'bool'        : types.boolean,
        'str'         : unicode_type,
    }
    non_none_nbtypes = [x for x in all_nbtypes.keys() if x != 'str']

    def from_d4p(self, spec):
        '''
        Import the raw daal4py spec and convert it to our needs
        '''
        assert any(x in spec for x in ['pyclass', 'c_name']), 'Missing required attribute in daal4py specification: ' + str(spec)
        assert 'attrs' in spec or any(x in spec for x in ['params', 'input_types']) or '__iface__' in spec['c_name'], 'invalid daal4py specification: ' + str(spec)
        if 'params' in spec:
            return D4PSpec(spec['pyclass'],
                           spec['c_name'],
                           params = spec['params'],
                           input_types = [(x[0], x[1].rstrip('*'), d4p_dtypes[x[3]]) for x in spec['input_types']],
                           result_dist =  d4p_dtypes[spec['result_dist']],
                           has_setup = spec['has_setup'])
        elif 'attrs' in spec:
            # filter out (do not support) properties for which we do not know the numba type
            attrs = []
            for x in spec['attrs']:
                typ = x[1].rstrip('*')
                if typ in self.all_nbtypes:
                    attrs.append((x[0], typ))
                #else:
                #    warnings.warn("Warning: couldn't find numba type for '" + x[1] +"'. Ignored.")
            return D4PSpec(spec['pyclass'],
                           spec['c_name'],
                           attrs = attrs)
        return None

    def __init__(self, spec): #algo, c_name, params, input_types, result_attrs, result_dist):
        '''
        See D4PSpec for input specification.
        Defines numba type. To make it usable also call activate().
        '''
        if 'alias' not in spec:
            self.mk_type(spec)
        else:
            self.name = spec['c_name']
        self.spec = spec


    def activate(self):
        '''Bring class to life'''
        if 'alias' in self.spec:
            return
        self.spec = self.from_d4p(self.spec)
        self.mk_ctor()
        self.mk_attrs()
        self.mk_boxing()

    def mk_type(self, spec):
        '''Make numba type and register opaque model'''
        assert 'pyclass' in spec, "Missing required attribute 'pyclass' in daal4py spec: " + str(spec)
        self.name = spec['pyclass'].__name__
        def mk_simple(name):
            class NbType(types.Opaque):
                '''Our numba type for given algo class'''
                def __init__(self):
                    super(NbType, self).__init__(name=name)
            return NbType

        # make type and type instance for algo, its result or possibly model
        # also register their opaque data model
        self.NbType = mk_simple(self.name + '_nbtype')
        _nbt = self.NbType()
        self.all_nbtypes[self.name] = _nbt
        # FIXME: fix after Numba #3372 is resolved
        setattr(types, 'd4p_'+self.name, _nbt)

        # register numba type so that numba knows about it (e.g.for args)
        @typeof_impl.register(spec['pyclass'])
        def typeof_me(val, c):
            return _nbt

        register_model(self.NbType)(models.OpaqueModel)

        # some of the classes can be parameters to others and have a default NULL/None
        # We need to cast Python None to C NULL
        exec(("@intrinsic\n"
              "def {name}_from_none(typingctx, obj):\n"
              "    def codegen(context, builder, sig, args):\n"
              "        zero = context.get_constant(types.intp, 0)\n"
              "        return builder.inttoptr(zero, context.get_value_type(algo_factory.all_nbtypes['{name}']))\n"
              "    return algo_factory.all_nbtypes['{name}'](types.none), codegen\n").format(name=self.name), globals())

    def mk_ctor(self):
        '''
        Declare type and lowering code for constructing an algo object
        Lowers algo's constructor: we just call the C-function.
        We provide an @intrinsic which calls the C-function and an @overload which calls the former.
        '''
        if not self.spec or not self.spec.params:
            if self.spec:
                # this must be a result or model, not an algo class
                # we don't want a constructor but Numba must know ts model for boxing/unboxing
                @typeof_impl.register(self.spec.pyclass)
                def typeof_index(val, c):
                    return self.all_nbtypes[self.name]

            return

        # TODO: check args

        # PR numba does not support kwargs when lowering/typing, so we need to fully expand arguments.
        # We can't do this with 'static' code because we do not know the argument names in advance,
        # they are provided in the D4PSpec. Hence we generate a function def as a string and python-exec it
        # The actual code is compiled lazily, which is probably why we cannot really bind variables here, we need to
        # expand everything to global names (hence the format(...) below). Has this changed?
        # What a drag.

        # see commit 1f502e4246b5c3df18a71df8348b1322210a149e for a true @overload
        # which however would need handling class hierachies of algorithms (e.g. optimization solvers, engines)
        cmm_string = ("@overload(daal4py.{name})\n"
                      "def mk_{name}_ovld({argsWdflt}):\n"
                      "    def mk_{name}({argsWdflt}):\n"
                      "        with objmode(r='d4p_{name}'):\n"
                      "            r = daal4py.{name}({args})\n"
                      "        return r\n"
                      "    return mk_{name}\n")
        cmm_string = cmm_string.format(name=self.name,
                                       cname=self.spec.c_name,
                                       cargs=', '.join(['*arr2C({})'.format(x[0]) if x[1] == 'table' else  x[0] for x in self.spec.params]),
                                       argsWdflt=', '.join([x[0] + ('=' + ('"{}"'.format(x[2]) if algo_factory.all_nbtypes[x[1]] == unicode_type else str(x[2]))
                                                                    if x[2] != None else '') for x in self.spec.params]),
                                       ityps=', '.join(["'{}'".format(x[1]) for x in self.spec.params]),
                                       precall ='\n        '.join(['{0} = unicode_to_std_str({0})'.format(x[0])
                                                                   if x[1] == 'str'
                                                                   else '{arg} = {arg} if {arg} != None else {typ}_from_none({arg})'.format(typ=x[1], arg=x[0])
                                                                   for x in self.spec.params if x[1] not in algo_factory.non_none_nbtypes]),
                                       postcall='\n        '.join(['del_str({0})'.format(x[0]) for x in self.spec.params if x[1] == 'str']),
                                       args=', '.join([x[0] for x in  self.spec.params]),
        )
        exec(cmm_string, globals(), {})


    def mk_attrs(self):
        '''
        Provide the typing and lowering for getters and calling compute
        Again, we can't use static code, so we need
        to generate code that we exec.
        '''

        if not self.spec:
            return

        # Initing the code we want to exec
        attr_code = ''
        ovl_code  = ''

        # The python name stub (module + class)
        name_stub = '_'+'.'.join([self.spec.pyclass.__module__.strip('_'), self.spec.pyclass.__name__])
        result_type = None

        # First handle properties (result/model classes)
        if self.spec.attrs:
            for a in self.spec.attrs:
                c_func = '_'.join(['get', self.spec.c_name, a[0]])

                # lowering code for properties through @overload_attribute
                attr_code += ("@overload_attribute(algo_factory.all_nbtypes['{0}'], '{1}')\n" # getters have no args (other than self)
                              "def {0}_get_{1}(obj):\n"
                              "  cfunc = types.ExternalFunction('{2}', {3}(algo_factory.all_nbtypes['{0}']))\n"
                              "  def _impl(obj):\n"
                              "    r = cfunc(obj)\n"
                              "    return {4}\n"
                              "  return _impl\n").format(self.name,
                                                         a[0],
                                                         c_func,
                                                         'types.voidptr' if a[1] in d4ptypes else a[1],
                                                         'nt2nd(r, {})'.format(d4ptypes[a[1]]) if a[1] in d4ptypes else 'r')

        # Now we handle compute and setup method of algorithm classes
        # we forward compute and setup to  the lower-level _compute method
        if self.spec.input_types:
            ovl_code += ("@overload_method(type(algo_factory.all_nbtypes['{name}']), '_compute')\n"
                         "def {name}_compute_setup(algo, {args}, setup):\n"
                         "    if isinstance(algo, type(algo_factory.all_nbtypes['{name}'])):\n"
                         "        ityps = [{ityps}]\n"
                         "        sig = [algo_factory.all_nbtypes['{name}']]\n"
                         "        for i in ityps:\n"
                         "            if i == 'data_or_file':\n"
                         "                sig += [types.voidptr, types.int64, types.int64, types.int64]\n"
                         "            else:\n"
                         "                sig.append(algo_factory.all_nbtypes[i])\n"
                         "        sig.append(types.boolean)\n"
                         "        sig = signature(algo_factory.all_nbtypes['{name}_result'], *sig)\n"
                         "        cfunc = types.ExternalFunction('compute_{cname}', sig)\n"
                         "\n"
                         "        def {name}_compute_setup_impl(algo, {args}, setup):\n"
                         "            return cfunc(algo, {cargs}, setup)\n"
                         "\n"
                         "        return {name}_compute_setup_impl\n"
                         "\n"
                         "@overload_method(type(algo_factory.all_nbtypes['{name}']), 'compute')\n"
                         "def {name}_compute(algo, {argsWdflt}):\n"
                         "    if isinstance(algo, type(algo_factory.all_nbtypes['{name}'])):\n"
                         "        def {name}_compute_impl(algo, {args}):\n"
                         "            return algo._compute({args}, False)\n"
                         "        return {name}_compute_impl\n")

            # Algo might have a setup call
            if self.spec.has_setup:
                ovl_code += ("@overload_method(type(algo_factory.all_nbtypes['{name}']), 'setup')\n"
                             "def {name}_setup(algo, {argsWdflt}):\n"
                             "    if {has_setup} and isinstance(algo, type(algo_factory.all_nbtypes['{name}'])):\n"
                             "        def {name}_setup_impl(algo, {args}):\n"
                             "            algo._compute({args}, True)\n"
                             "            return\n"
                             "        return {name}_setup_impl\n")

            ovl_code = ovl_code.format(name=self.name,
                                       cname=self.spec.c_name,
                                       args=', '.join([x[0] for x in self.spec.input_types]),
                                       argsWdflt=', '.join(['{}=None'.format(x[0]) for x in self.spec.input_types]),
                                       ityps=', '.join(["'{}'".format(x[1]) for x in self.spec.input_types]),
                                       cargs=', '.join(['*inp2d4p({0})'.format(x[0]) if x[1] == 'data_or_file' else '{}'.format(x[0]) for x in self.spec.input_types]),
                                       has_setup=self.spec.has_setup)
        # endif self.spec.input_types

        try:
            exec(attr_code+ovl_code, globals(), {})
        except Exception as e:
            import sys
            print("Unexpected error:", sys.exc_info()[0])
            raise


    def cy_unboxer(self):
        '''
        Return the address of the cython generated C(++) function which
        provides the actual DAAL C++ pointer for our result/model Python object.
        '''
        addr = get_cython_function_address("_daal4py", 'unbox_'+self.spec.c_name)
        return addr


    def mk_boxing(self):
        '''provide boxing and unboxing'''
        if not self.spec:
            return

        ll.add_symbol('unbox_'+self.spec.c_name, self.cy_unboxer())

        @unbox(self.NbType)
        def unbox_me(typ, obj, c):
            'Call C++ unboxing function and return as NativeValue'
            # Define the signature
            fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(),])
            # Get function
            fn = c.builder.module.get_or_insert_function(fnty, name='unbox_'+self.spec.c_name)
            # finally generate the call
            ptr = c.builder.call(fn, [obj])
            return NativeValue(ptr, is_error=c.pyapi.c_api_error())
        
        @box(self.NbType)
        def box_me(typ, val, c):
            'Call staticmethod __from_ptr function with the C++ pointer, returning Python object.'
            ll_intp = c.context.get_value_type(types.uintp)
            addr = c.builder.ptrtoint(val, ll_intp)
            v = c.box(types.uintp, addr)
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(self.spec.pyclass))
            res = c.pyapi.call_method(class_obj, "__from_ptr", (v,))
            return res


##############################################################################
##############################################################################
##############################################################################
# finally let the factory do its job

open_daal4py()

# first define types
algos = [algo_factory(x) for x in hpat_spec]

# then setup aliases
for s in hpat_spec:
    if 'alias' in s:
        # for assume we have no recurring aliasing
        assert s['alias'] in algo_factory.all_nbtypes, "Recurring aliasing not supported"
        algo_factory.all_nbtypes[s['c_name']] =  algo_factory.all_nbtypes[s['alias']]
# now bring life to the classes
algo_map = {}
for a in algos:
    a.activate()
    algo_map[algo_factory.all_nbtypes[a.name]] = a

@overload(daal4py.daalinit)
def ovl_daalinit():
    c_daalinit = types.ExternalFunction('c_daalinit', types.void(types.int32))
    def daalinit():
        c_daalinit()
    return daalinit

@overload(daal4py.daalfini)
def ovl_daalfini():
    c_daalfini = types.ExternalFunction('c_daalfini', types.void())
    def daalfini():
        c_daalfini()
    return daalfini

@overload(daal4py.num_threads)
def ovl_num_threads():
    c_num_threads = types.ExternalFunction('c_num_threads', types.uint64())
    def num_threads():
        return c_num_threads()
    return num_threads

@overload(daal4py.num_procs)
def ovl_num_procs():
    c_num_procs = types.ExternalFunction('c_num_procs', types.uint64())
    def num_procs():
        return c_num_procs()
    return num_procs

@overload(daal4py.my_procid)
def ovl_my_procid():
    c_my_procid = types.ExternalFunction('c_my_procid', types.uint64())
    def my_procid():
        return c_my_procid()
    return my_procid

#del algos

##############################################################################
##############################################################################

from hpat.distributed_analysis import DistributedAnalysis, Distribution as DType

def _analyze_call_d4p():
    '''
    Analyze distribution for calls to daal4py.
    Return True of a call for daal4py was detected and handled.
    We cannot simply "meet" distributions, the d4p algos accept a certain decomposition only.
    The required distribution/decomposition is defined in the algorithms specs.
    We raise an exception if the required distribution cannot be met.
    '''

    def _analyze(lhs, func_mod, mod_typ, func_name, args, array_dists):
        assert func_name == 'compute' and mod_typ in algo_map, "unexpected type/function to daal4py function-call analysis."
        # every d4p algo gets executed by invoking "compute".
        # we need to find the algorithm that's currently called
        algo = algo_map[mod_typ]
        rdist = algo.spec.result_dist if algo.spec.result_dist != None else DType.OneD
        # handle all input arguments and set their distribution as given by the spec
        for i in range(len(args)):
            aname = args[i].name
            adist = algo.spec.input_types[i][2]
            if aname not in array_dists:
                array_dists[aname] = adist
            else:
                if algo.spec.result_dist:
                    min_adist = DType.OneD_Var if adist == DType.OneD else adist
                else:
                    # if algo has no distributed mode, we just follow
                    min_adist = DType.REP
                assert array_dists[aname].value <= DType.OneD.value, "Cannot handle unknown distribution type"
                # bail out if there is a distribution conflict with some other use of the argument
                # FIXME: handle DType.Thread and Disribution.REP as equivalent
                assert array_dists[aname].value >= min_adist.value,\
                       'Distribution of argument {} ({}) to "daal4py.{}.compute" must be "{}". '\
                       'Some other use of it demands "{}", though.'\
                       .format(i+1, algo.spec.input_types[i][0], algo.name, adist, array_dists[aname])
        # handle distribution of the result
        if lhs not in array_dists:
            array_dists[lhs] = rdist
        else:
            array_dists[lhs] = DType(min(array_dists[lhs].value, rdist.value))
            if algo.spec.result_dist:
                min_rdist = DType.OneD_Var if rdist == DType.OneD else algo.spec.result_dist
            else:
                # if algo has no distributed mode, the result dist follows the input
                min_rdist = DType.REP
            assert array_dists[lhs].value >= min_rdist.value,\
                'Distribution ({}) to "daal4py.{}.compute" must be at least "{}". '\
                'Some other use of it demands "{}", though.'\
                .format(algo.name, min_rdist, array_dists[lhs])
        return True

    for a in algo_map:
        DistributedAnalysis.add_call_analysis(a, 'compute', _analyze)
