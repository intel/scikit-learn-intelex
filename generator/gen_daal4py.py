#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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

import glob, os, re
from pprint import pformat, pprint
from os.path import join as jp
from collections import defaultdict, OrderedDict
from .parse import parse_header
from jinja2 import Template
from .wrappers import required, ignore, defaults, specialized, has_dist, ifaces
from .wrapper_gen import wrapper_gen, typemap_wrapper_template

try:
    basestring
except NameError:
    basestring = str

def first_non_default(value):
    """
    Returns first element in value which does not contain "default".
    If value is/has only a single element it is returned in any case.
    Otherwise assumes a list/tuple contains at least one non-default element.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        for x in value:
            if 'default' not in x:
                return x
        print(value)
        assert False
    return value

def cpp2hl(cls):
    return cls.replace('::', '_')

###############################################################################
# Managing SWIG interface files (jinja templates).
# - Uses parse.py to read in C++ headers files.
# - reads in an existing config for each namespace and compares it to what it
#   found in the C++ headers
# - Converts resulting data strucutures into SWIG consumable files
#   this requries some recursive searching and expansion
###############################################################################

###############################################################################
def cleanup_ns(fname, ns):
    """return a sanitized namespace name"""
    # strip of namespace 'interface1'
    while len(ns) and ns[-1].startswith('interface'):
        del ns[-1]
    # cleanup duplicates
    while len(ns) >= 2 and ns[-1] == ns[len(ns)-2]:
        del ns[-1]
    # we should now have our namespace hierachy
    if len(ns) == 0 or ns[0] != 'daal':
        print(fname + ":0: Warning: No namespace (starting with daal) found in " + fname + '. Ignored.')
        return False
    nsn = '::'.join(ns[1:])
    # namespace 'daal' is special, it's empty
    if len(nsn) == 0 :
        nsn = 'daal'
    # Multiple namespaces will leave 'interface' in the hierachy
    # we cannot handle these cases
    if 'interface' in nsn:
        print(fname + ":0: Warning: Multiple namespaces found in " + fname + '. Ignored.')
        return False
    return nsn


###############################################################################
###############################################################################
def splitns(x):
    tmp_ = x.rsplit('::', 1)
    if len(tmp_) == 1:
        return ('', x)
    else:
        return tmp_

def get_parent(ns):
    tmp = ns.rsplit('::', 1)
    return tmp[0] if len(tmp) > 1 else 'daal'

###############################################################################
###############################################################################
def compare(ns, a, b, allowed_diffs, ignores):
    """
    Helper function comparing two configs.
    Iterating through all keys in config a and checks if the same exists in b.
    Certain fields are skipped.
    A Warning message will be printed for each difference found.
    """
    cfga = a[0]
    cfgb = b[0]
    # go through all entries in a
    for x in cfga:
        if x not in ['package', 'module', 'namespace', 'deps', 'renames', 'computes', 'ignore',]:
            if not cfga[x]:
                continue
            # check if in b
            if x in cfgb:
                # go through all sub-entries
                for y in cfga[x]:
                    found = False
                    # we need to account for our special '!' syntax for jinja processing
                    for yy in ([y, y.strip('!'), y+'!', '!'+y] if isinstance(y, basestring) else [y]):
                        if yy in cfgb[x]:
                            if isinstance(cfga[x],dict) and (not isinstance(cfgb[x],dict) or cfga[x][y] != cfgb[x][yy]):
                                if '::'.join([ns, x, y.strip('!')]) not in allowed_diffs:
                                    # value is not identical and it's not in allowed diffs
                                    print(b[1] + ':0: Warning: ' + x + '->' + y + ': ' + str(cfga[x][y]) + ' differs')
                                    print(a[1] + ':0: Warning: from what is defined here')
                            found = True
                            break;
                    if not found and y not in ignores and '::'.join([ns, x, str(y)]) not in allowed_diffs:
                        # key not found in b
                        print(b[1] + ':0: Warning: ' + x + '->' + str(y) + ' not found for ' + ns)
                        print(a[1] + ':0: Warning: but it is defined here')
            else:
                if '::'.join([ns, x]) not in allowed_diffs:
                    # key not found in b and not in allowed diffs
                    print(b[1] + ':0: Warning: ' + x + ' not found for ' + ns)
                    print(a[1] + ':0: Warning: but it is defined here')

###############################################################################

###############################################################################
class namespace(object):
    """Holds all extracted data of a namespace"""
    def __init__(self, name):
        self.classes = {}
        self.enums = {}
        self.typedefs = {}
        self.headers = []
        self.includes = set()
        self.name = name
        self.need_methods = False
        self.steps = set()
        self.children = set()


###############################################################################
    def resolve_methods(self, ns, namespace_dict):
        """
        Find and return first list of Method enum in namespace hierachy.
        Getting called assumes the ns needs methods and so will find in parents
        if there are none in the ns itself.
        """
        if 'Method' in self.enums:
            # this ns has its own definition of methods
            pfx = '' if ns == self.name else 'daal::' + self.name + '::'

            rev = defaultdict(list)
            for m in self.enums['Method']:
                rev[self.enums['Method'][m]].append(pfx + m)
            res = sorted(rev[''])
            del rev['']
            for m in rev:
                if m.isdigit():
                    res.append(sorted(rev[m]) if len(rev[m]) > 1 else rev[m][0])
                elif 'default' in m:
                    res.append(sorted(list(rev[m]) + [pfx + m]))
                else:
                    print(rev)
                    assert False
            return list(res)

        # we do not have methods in this namespace
        # let's find them in our parents
        split_ns = splitns(self.name)[0]
        if len(split_ns) > 1 and split_ns[0] in namespace_dict:
            return namespace_dict[split_ns[0]].resolve_methods(ns, namespace_dict)
        else:
            return []


###############################################################################
    def resolve_deps(self, namespace_dict, deps):
        """Performs one recursive pass through dependences from #include directives.
           Adds dependent namespaces to its list of deps and return it."""
        # let's determine which namespace have the include files a given ns needs to #define as USE_*
        #   also determine the interface files to be imported
        for inc in self.includes:
            # check which namespace the #included file belongs to
            for ns2 in namespace_dict:
                if inc in namespace_dict[ns2].headers:
                    if ns2 not in ['daal', 'algorithms', 'services', 'data_management', self.name]:
                        # we found the namespace -> add once
                        deps[self.name].add(ns2)
                        # we also add all the deps that
                        for n in deps[ns2]:
                            deps[self.name].add(n)
                    break
        return deps[self.name]


###############################################################################
    def resolve_computes(self, namespace_dict):
        """
        Determine which compute wrappers are needed and return dict.
        The values are strings with the standard jinja macro calls.
        """
        computes = {}
        # Determine the compute() wrappers we need
        #  (e.g. if there are Batch, Online, Distributed algorithms)
        if 'Batch' in self.classes:
            computes['Batch'] = '{{add_compute("Batch")}}'
        if 'Online' in self.classes:
            computes['Online'] = '{{add_compute2(ns, cfg, "Online")}}'
        if 'Distributed' in self.classes:
            computes['Distributed'] = '{{add_compute2(ns, cfg, "Distributed")}}'
        return computes


###############################################################################
    def format_setgets(self):
        """
        Returns a list of strings with SWIG %rename directives, one for each set/get method.
        The new name is generated from the argument types.
        """
        rs = []
        for c in self.classes:
            if len(self.classes[c].setgets) and isinstance(self.classes[c].setgets, (list, tuple)):
                for r in self.classes[c].setgets:
                    rs.append('%rename(' + r[0] + r[2].replace('Id', '') + ') /*' + r[1] + '*/ daal::{{ns}}::interface1::' + c + '::' + r[3])
        return rs


###############################################################################
    def as_iface(self, namespace_dict, deps):
        """returns a dict in the iface-tmpl format for our i.tmpl files"""
        cfg = {
            'classes'   : [c for c in self.classes if not self.classes[c].template_args],
            'computes'  : self.resolve_computes(namespace_dict),
            'deps'      : list(deps),
            'includes'  : self.headers,
            'module'    : splitns(self.name)[0],
            'namespace' : self.name,
            'package'   : splitns(self.name)[0].replace('::', '.'),
            'renames'   : self.format_setgets(),
            'steps'     : ['daal::' + x for x in self.steps],
            'templates' : {c: self.classes[c].template_args for c in self.classes if self.classes[c].template_args and not self.classes[c].partial},
        }
        for c in self.classes:
            cfg['templates'].update({f[0]: f[1] for f in self.classes[c].templates})
        if self.need_methods:
            cfg['methods'] = self.resolve_methods(self.name, namespace_dict)
            if not len(cfg['methods']):
                print(self.name)
                assert False
            # if the methods are from a parent, we assume they are all from the same ns.
            # we need to prepend the parent's fully qualified namespace to each
            # method template parameter.
            if '::' in cfg['methods'][0]:
                tmp = splitns(cfg['methods'][0])[0]
                for t in cfg['templates']:
                    for a in cfg['templates'][t]:
                        if a[0] == 'method' and len(a[2]):
                            a[2] = tmp + '::' + a[2]
        # template default values get resolved to the first non-default-name of the same
        # enum value
        for tmpl in cfg['templates']:
            for tmplarg in cfg['templates'][tmpl]:
                if len(tmplarg) >= 3 and tmplarg[1] in cfg and 'default' in tmplarg[2]:
                    for val in cfg[tmplarg[1]]:
                        if tmplarg[2] == val:
                            break
                        if isinstance(val, (tuple, list)) and tmplarg[2] in val:
                            tmplarg[2] = first_non_default(val)
                            break

        d = []
        for k in cfg:
            if len(cfg[k]) == 0:
                d.append(k)
        for k in d:
            del cfg[k]

        return cfg


###############################################################################
    def from_iface(self, fname):
        if not os.path.isfile(fname):
            return None
        cfgstr = None
        with open(fname, "r") as f:
            for l in f:
                if cfgstr != None:
                    if '%}' in l:
                        cfgstr = cfgstr.strip()
                        if cfgstr[0] != '{':
                            cfgstr = '{' + cfgstr
                        if '}' in l.replace('%}', ''):
                            cfgstr += '}'
                        return eval(cfgstr, {'fptypes': 'fptypes', 'cmodes': 'cmodes', 'ntypes': 'ntypes', 'stypes': 'stypes'})
                    else:
                        cfgstr += l
                elif '{% set cfg =' in l:
                    cfgstr = ''
        return None


###############################################################################
    def is_empty(self):
        return not any(len(x) > 0 for x in [self.classes, self.enums, self.steps])


###############################################################################
    def write(self, cfg, fname):
        #print("Writing " + fname)
        with open(fname, 'w') as template_file:
            template_file.write('cfg =\n')
            pprint(cfg, stream=template_file, width=110)

###############################################################################
    def write_update(self, cfg, fname):
        if not os.path.isfile(fname):
            return
        #print("Updating " + fname)
        cfgstr = False
        o = ''
        with open(fname, "r") as f:
            for l in f:
                if cfgstr:
                    if '%}' in l:
                        o += '{% set cfg =\n' + pformat(cfg, width=110) + '\n%}\n\n'
                        cfgstr = False
                elif '{% set cfg =' in l:
                    cfgstr = True
                else:
                    o += l
        with open(fname, 'w') as f:
            f.write(o)

###############################################################################
###############################################################################
class cython_interface(object):
    """collecting and generating data for SWIG"""

    # classes/functions we generally ignore
    ignores = ['AlgorithmContainerIface', 'AnalysisContainerIface',
               'PredictionContainerIface', 'TrainingContainerIface', 'DistributedPredictionContainerIface',
               'BatchContainerIface', 'OnlineContainerIface', 'DistributedContainerIface',
               'BatchContainer', 'OnlineContainer', 'DistributedContainer',
               'serializeImpl', 'deserializeImpl', 'serialImpl',
               'getEpsilonVal', 'getMinVal', 'getMaxVal', 'getPMMLNumType', 'getInternalNumType', 'getIndexNumType',
               'allocateNumericTableImpl', 'allocateImpl',
               'setPartialResultStorage', 'addPartialResultStorage',]

    # file we ignore/skip
    ignore_files = ['daal_shared_ptr.h', 'daal.h', 'daal_win.h', 'algorithm_base_mode_batch.h',
                    'algorithm_base.h', 'algorithm.h', 'ridge_regression_types.h', 'kdtree_knn_classification_types.h',
                    'multinomial_naive_bayes_types.h', 'daal_kernel_defines.h', 'linear_regression_types.h',
                    'multi_class_classifier_types.h']

    # allowed diffs
    # matched against $namespace::$symbol; $symbol can be be a class, template or key in our config
    allowed_diffs = ['algorithms::implicit_als::training::init::templates::Distributed',
                     'algorithms::implicit_als::training::templates::Distributed',
                     'algorithms::implicit_als::training::templates::Batch',
                     'algorithms::multi_class_classifier::prediction::templates::Batch',
                     'algorithms::implicit_als::training::stages',
                     'algorithms::implicit_als::training::dmethods',
                     'algorithms::implicit_als::training::bmethods',
                     'algorithms::kmeans::init::templates::Distributed',
                     'algorithms::kmeans::init::s1methods',
                     'algorithms::kmeans::init::s2mmethods',
                     'algorithms::kmeans::init::s2l34methods',
                     'algorithms::kmeans::init::s5methods',]

    ignore_ns = ['daal', 'algorithms', 'services', 'data_management']

    # default value indicators
    defaults = {'double': 'NaN64',
                'float': 'NaN32',
                'int': '-1',
                'long': '-1',
                'size_t': '-1',
                'bool': 'False',
                #'std::string' : '""',
                'std::string &' : '""',
                'daal::data_management::NumericTablePtr': 'None', # 'data_management::NumericTablePtr()',
            }
    defaults.update({v: 'None' for v in ifaces.values()}) #v+'()

    done = []


###############################################################################
    def __init__(self, include_root):
        self.include_root = include_root
        self.namespace_dict = defaultdict(namespace)


###############################################################################
    def read(self):
        """
        Walk through each directory in the root dir and read in C++ headers.
        Creating a namespace dictionary. Of course, the it needs to go through every header file to find out
        what namespace it is affiliated with. Once it does this, we have a dictionary where the key is the namespace
        and the values are namespace class objects. These objects carry all information as extracted by parse.py.
        """
        for (dirpath, dirnames, filenames) in os.walk(self.include_root):
            for filename in filenames:
                if filename.endswith('.h') and not any(filename.endswith(x) for x in cython_interface.ignore_files):
                    fname = jp(dirpath,filename)
                    with open(fname, "r") as header:
                        parsed_data = parse_header(header, cython_interface.ignores)

                    ns = cleanup_ns(fname, parsed_data['ns'])
                    if ns:
                        if ns not in self.namespace_dict:
                            self.namespace_dict[ns] = namespace(ns)
                        pns = get_parent(ns)
                        if pns not in self.namespace_dict:
                            self.namespace_dict[pns] = namespace(pns)
                        if ns != 'daal':
                            self.namespace_dict[pns].children.add(ns)
                        self.namespace_dict[ns].includes = self.namespace_dict[ns].includes.union(parsed_data['includes'])
                        self.namespace_dict[ns].steps = self.namespace_dict[ns].steps.union(parsed_data['steps'])
                        self.namespace_dict[ns].classes.update(parsed_data['classes'])
                        self.namespace_dict[ns].enums.update(parsed_data['enums'])
                        self.namespace_dict[ns].typedefs.update(parsed_data['typedefs'])
                        self.namespace_dict[ns].headers.append(fname.replace(self.include_root, '').lstrip('/'))
                        if parsed_data['need_methods']:
                            self.namespace_dict[ns].need_methods = True


###############################################################################
    def digest(self):
        """
        1. Process raw data in our namespace_dict.
        """
        # sort include files to put *_types file first in list
        for ns in self.namespace_dict:
            self.namespace_dict[ns].headers.sort(key=lambda x: x if '_types' not in x else '_')



###############################################################################
    def write_and_compare(self, newdir, olddir, update=None):
        """
        Write the extract data into files with extension as newdir/*.i.tmpl.new.
        Then compare with corresponding file in dir "olddir".
        """
        # let's recursively resolve all dependences between namespaces
        all_deps = defaultdict(set)
        added = 1
        while added > 0:
            added = 0
            for ns in self.namespace_dict:
                oldcnt = len(all_deps[ns])
                all_deps[ns] = self.namespace_dict[ns].resolve_deps(self.namespace_dict, all_deps)
                added += len(all_deps[ns]) - oldcnt
        for ns in self.namespace_dict:
            nso = self.namespace_dict[ns]
            newcfg = nso.as_iface(self.namespace_dict, all_deps[ns])
            newfname = jp(newdir, ns.replace('::', '__') + '.i.tmpl.new')
            nso.write(newcfg, newfname)
            if ns in cython_interface.ignore_ns:
                continue
            oldfname = jp(olddir, ns.replace('::', '__') + '.i.tmpl')
            oldcfg = nso.from_iface(oldfname)
            if oldcfg:
                # compare both ways
                compare(ns, (newcfg, newfname), (oldcfg, oldfname), cython_interface.allowed_diffs, [])
                compare(ns, (oldcfg, oldfname), (newcfg, newfname), cython_interface.allowed_diffs, cython_interface.ignores)
            elif not nso.is_empty():
                print('Error: Could not find file ' + oldfname + ' or config in file.')
            if update and update in newcfg:
                if oldcfg:
                    u = {}
                    u[update] = newcfg[update]
                    oldcfg.update(u)
                    nso.write_update(oldcfg, oldfname)
                else:
                    print('Warning: no config for ' + oldfname)



###############################################################################
###############################################################################
# HLAPI/postprocessing starts here

###############################################################################
    def get_ns(self, ns, c, attrs=['classes', 'enums', 'typedefs']):
        """
        Find class c starting in namespace ns.
        c can have qualified namespaces in its name.
        We go up to all enclosing namespaces of ns until we find c (e.g. current-namespace::c)
        or we reached the global namespace.
        """
        # we need to cut off leading daal::
        if c.startswith('daal::'):
            c = c[6:]
        tmp = c.split('::')
        tmp = splitns(c)
        cns = ('::' + tmp[0]) if len(tmp) == 2 else '::' # the namespace-part of our class
        cname = tmp[-1]                                  # class name (stripped off namespace)
        currns = ns + cns   # current namespace in which we look for c
        done = False
        while currns and not done:
            # if in the outmost level we only have cns, which starts with '::'
            tmpns = currns.strip(':')
            if tmpns in self.namespace_dict and any(cname in getattr(self.namespace_dict[tmpns], a) for a in attrs):
                return tmpns
            if ns == 'daal':
                done = True
            else:
                if currns.startswith('::'):
                    ns = 'daal'
                else:
                    ns = splitns(ns)[0]
                currns = ns + cns
        return None


###############################################################################
    def get_all_attrs(self, ns, cls, attr):
        """
        Return an ordered dict, combining the 'attr' dicts of class 'cls' and all its parents.
        Note: this looks for parents of 'cls' not parents of 'ns'!
        """
        # we need to cut off leading daal::
        cls = cls.replace('daal::', '')
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes:
            if ns in self.namespace_dict and '::' not in cls:
                # the class might be in one of the parent ns
                ns = self.get_ns(ns, cls)
                if ns == None:
                    return None
            else:
                return None

        pmembers = OrderedDict()
        # we add ours first. When expanding, duplicates from parents will be ignored.
        tmp = getattr(self.namespace_dict[ns].classes[cls], attr)
        for a in tmp:
            if '::' in a:
                pmembers[a] = tmp[a]
            else:
                pmembers[ns + '::' + a] = tmp[a]
        for parent in self.namespace_dict[ns].classes[cls].parent:
            parentclass = cls
            pns = ns
            # we need to cut off leading daal::
            sanep = parent.split()[-1].replace('daal::', '')
            parentclass = splitns(sanep)[1]
            pns = self.get_ns(pns, sanep)
            if pns != None:
                pmembers.update(self.get_all_attrs(pns, parentclass, attr))
        return pmembers


###############################################################################
    def to_hltype(self, ns, t):
        """
        Return triplet (type, {'stdtype'|'enum'|'tm'|'class'|'?'}, namespace) to be used in the interface
        for given type 't'.
        Returns 'tm' if a SWIG typemap exists.
        For classes, we also add lookups in namespaces that DAAL C++ API finds through "using".
        """
        tns, tname = splitns(t)
        if t in ['double', 'float', 'int', 'size_t']:
            return (t, 'stdtype', '')
        if t in ['bool']:
            return ('bool', 'stdtype', '')
        if t.endswith('ModelPtr'):
            thens = self.get_ns(ns, t, attrs=['typedefs'])
            return ('daal::' + thens + '::ModelPtr', 'tm', tns)
        if t in ['data_management::NumericTablePtr',] or t in ifaces.values():
            return ('daal::' + t, 'tm', tns)
        tt = re.sub(r'(?<!daal::)services::SharedPtr', r'daal::services::SharedPtr', t)
        tt = re.sub(r'(?<!daal::)algorithms::', r'daal::algorithms::', tt)
        if tt in ifaces.values():
            return (tt, 'tm', tns)
        tns = self.get_ns(ns, t)
        if tns:
            tt = tns + '::' + tname
            if tt == t:
                return ('std::string &', 'enum', tns) if tname in self.namespace_dict[tns].enums else (tname, 'class', tns)
            else:
                return self.to_hltype(ns, tt)
        else:
            usings = ['algorithms::optimization_solver']
            if not any(t.startswith(x) for x in usings):
                for nsx in usings:
                    r = self.to_hltype(ns, nsx + '::' + t)
                    if r:
                        return r
        return None if '::' in t else (t, '??', '??')


###############################################################################
    def get_values(self, ns, n):
        """
        Return available values for given enum or special "group".
        """
        if n == 'fptypes':
            return ['double', 'float']
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return [nns + '::' + x for x in self.namespace_dict[nns].enums[nn]]
            return ['unknown_' + nns + '_class_'+n]
        return ['unknown_'+n]


###############################################################################
    def get_tmplarg(self, ns, n):
        """
        Return template argument specifier: "typename" or actual type.
        """
        if n == 'fptypes':
            return 'typename'
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return nns + '::' + nn
            return 'unknown_' + nns + '_class_'+n
        return 'unknown_'+n


###############################################################################
    def get_class_for_typedef(self, ns, cls, td):
        """
        Find the Result type for given algorithm in the C++ namespace hierachy.
        Strips off potential SharedPtr def.
        """
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes or td not in self.namespace_dict[ns].classes[cls].typedefs:
            return None

        res =  self.namespace_dict[ns].classes[cls].typedefs[td]
        tmp = splitns(res)
        if res.endswith('Type'):
            # this is a dirty hack: we assume there are no typedefs *Type outside classes
            assert res.endswith(td)
            ret = self.get_class_for_typedef(self.get_ns(ns, tmp[0]), splitns(tmp[0])[1], td)
        else:
            ret = (self.get_ns(ns, res.split('<')[0]), tmp[1])
        return ret
        # if not res:
        #     assert False
        #     for parent in self.namespace_dict[ns].classes[cls].parent:
        #         pns = ns
        #         # we need to cut off leading daal::
        #         sanep = parent.split()[-1].replace('daal::', '')
        #         parentclass = splitns(sanep)[1]
        #         pns = self.get_ns(pns, sanep)
        #         res = self.get_result(pns, parentclass)
        #         if res:
        #             return res
        #     return None
        # m = re.match(r'.+SharedPtr<\s*((\w|::)+)>.*', res[0])
        # if m:
        #     rclass = m.group(1)
        #     assert False
        # else:
        #     rclass = res[0]


###############################################################################
    def get_expand_attrs(self, ns, cls, attr):
        """
        Find enum type for attributes in "attr" dict of class "cls" and returns
        2-tuple of lists of tuples
          [0] list of members it can map to a hltype (namespace, member/value name, type of attr)
          [1] list of members it can not map tp hltype (namespace, attribute)
        Only standard data types and those with typemaps can be mapped to a hltype.
        """
        attrs = self.get_all_attrs(ns, cls, attr)
        explist = []
        ignlist = []
        for i in attrs:
            inp = splitns(i)[1]
            ins = self.get_ns(ns, i)
            assert ins
            assert ins in self.namespace_dict
            assert inp in self.namespace_dict[ins].enums
            hlt = self.to_hltype(ns, attrs[i])
            if hlt:
                if hlt[1] in ['stdtype', 'enum', 'tm']:
                    for e in self.namespace_dict[ins].enums[inp]:
                        if not any(e in x for x in explist):
                            explist.append((ins, e, hlt[0]))
                else:
                    print("// Warning: ignoring " + ns + " " + str(hlt))
                    ignlist.append((ins, i))
            else:
                print("// Warning: could not find hlt for " + ns + ' ' + cls + " " + i + " " + attrs[i])
        return (explist, ignlist)


###############################################################################
    def prepare_resultmaps(self, ns):
        """
        Prepare info about typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
        Looks up return type and then target-language independently creates lists of its content.
        """
        jparams = {}
        res = self.get_class_for_typedef(ns, 'Batch', 'ResultType')
        if not res and 'Result' in self.namespace_dict[ns].classes:
            res = (ns, 'Result')
        if res and '_'.join(res) not in self.done:
            self.done.append('_'.join(res))
            attrs = self.get_expand_attrs(res[0], res[1], 'sets')
            if attrs and attrs[0]:
                jparams = {'class_type': 'daal::' + res[0] + '::' + res[1] + 'Ptr',
                           'enum_gets': attrs[0],
                           'named_gets': [],
                       }
            else:
                print('// Warning: could not determine Result attributes for ' + ns)
        elif res:
            jparams = {'class_type': 'daal::' + res[0] + '::' + res[1] + 'Ptr',}
        else:
            print('// Warning: no result found for ' + ns)
        return jparams


###############################################################################
    def prepare_modelmaps(self, ns, mname='Model'):
        """
        Return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        """
        jparams = {}
        if mname in self.namespace_dict[ns].classes:
            model = self.namespace_dict[ns].classes[mname]
            jparams = {'class_type': 'daal::' + ns + '::ModelPtr',
                       'enum_gets': [],
                       'named_gets': [],
                   }
            huhu = self.get_all_attrs(ns, mname, 'gets')
            for g in huhu:
                if not any(g.endswith(x) for x in ['SerializationTag',]):
                    gn = splitns(g)[1].replace('get', '')
                    if not any(gn == x[1] for x in jparams['named_gets']):
                        jparams['named_gets'].append((huhu[g], gn))
        return jparams


###############################################################################
    def expand_typedefs(self, ns):
        """
        We expand all typedefs in classes/namespaces wihtout recursing
        to outer scopes or namespaces.
        """
        def expand_td(typedefs):
            done = 1
            while done != 0:
                done = 0
                for td1 in typedefs:
                    for td2 in typedefs:
                        if td1 != td2 and (td1+'::') in typedefs[td2]:
                            typedefs[td2] = typedefs[td2].replace(td1, typedefs[td1])
                            done += 1

        expand_td(self.namespace_dict[ns].typedefs)
        for c in self.namespace_dict[ns].classes:
            expand_td(self.namespace_dict[ns].classes[c].typedefs)


###############################################################################
    def prepare_hlwrapper(self, ns, mode, func):
        """
        Prepare data structures for generating high level wrappers.

        This is main function for generating high level wrappers.
        Here we prepare and return the data structure from which wrappers can be generated.
        The information is extracted from self.namespace_dict.

        We first extract template parameters and setup a generic structure/array. The generated array
        holds the general template spec in entry [0] and specializations in the following entires (if exist).

        We then extract input arguments. The resulting arrays begin with required inputs followed by optional inputs.
        We get the input arguments by expanding the enums from the set methods in the Input class (InputType).
        Each value in the enum becomes a separate input arguments.

        Next we extract parameters - required and optional separately. Each member of parameter_type becomes
        a separate arguments. Default values are handled in the lower C++ level. We provide "generic" default
        values like -1, NULL and "" for *all* parameters. These values cannot be reasonably used by the user.

        Next we extract type-map (native_type) information for Model and Result types.
        """
        if mode in self.namespace_dict[ns].classes:
            ins = splitns(self.namespace_dict[ns].classes[mode].typedefs['InputType'])[0] if 'InputType' in self.namespace_dict[ns].classes[mode].typedefs else ns
            jparams = {'ns': ns,
                       'algo': func,
                       'template_decl': OrderedDict(),
                       'template_spec': [],
                       'template_args': [],
                       'params_opt': OrderedDict(),
                       'params_req': OrderedDict(),
                       'input_args': [],
                       's1': 'step1Local',
                       's2': 'step2Master',
                   }
            # at this point required parameters need to be explicitly/maually provided in wrappers.required
            if ns in required and mode in required[ns]:
                jparams['params_req'] = OrderedDict(required[ns][mode])
            f = ''
            tdecl = []
            if self.namespace_dict[ns].classes[mode].template_args:
                if ns in specialized and mode in specialized[ns]:
                    # there might be specialized template classes, we provide explicit specs
                    v = specialized[ns][mode]
                    tdecl.append({'template_decl': v['tmpl_decl'],
                                  'template_args': None,
                                  'pargs': None})
                    for s in v['specs']:
                        tdecl.append({'template_decl': OrderedDict([(x, v['tmpl_decl'][x]) for x in s['template_decl']]),
                                      'template_args': [s['expl'][x] if x in s['expl'] else x for x in v['tmpl_decl']],
                                      'pargs': [s['expl'][x] for x in s['expl']]})
                else:
                    # 'normal' template
                    tdecl.append({'template_decl': OrderedDict([(t[0], {'template_decl': self.get_tmplarg(ns, t[1]),
                                                                        'values': self.get_values(ns, t[1]),
                                                                        'default': t[2].replace('DAAL_ALGORITHM_FP_TYPE', 'double')}) for t in self.namespace_dict[ns].classes[mode].template_args]),
                                  'template_args': [t[0] for t in self.namespace_dict[ns].classes[mode].template_args],
                                  'pargs': None})

            # A parameter can be a specialized template. Sigh.
            # we need to provide specialized template classes for them
            # at some point we might have other things that influence this (like result or input).
            # for now, we only check for Parameter specializations
            decl_opt, decl_req , call_opt, call_req = [], [], [], []
            for td in tdecl:
                if 'template_args' in td:
                    # this is a "real" template for which we need a body
                    td['params_req'] = OrderedDict()
                    td['params_opt'] = OrderedDict()
                    pargs_exp = ','.join([splitns(x)[1] for x in td['pargs']]) if td['pargs'] else ''
                    cls = mode + pargs_exp
                    p = self.get_all_attrs(ns, cls, 'members')
                    if p and any(x.endswith('parameter') for x in p):
                        if 'ParameterType' in self.namespace_dict[ns].classes[cls].typedefs:
                            p = self.get_class_for_typedef(ns, cls, 'ParameterType')
                            if td['pargs'] != None:
                                p = (p[0], p1[1] + pargs_exp)
                            parms = self.get_all_attrs(p[0], p[1], 'members') if p else None
                            if not parms:
                                print('// Warning: no members of "parameter" found for ' + str(p))
                                continue
                        else:
                            print(' '.join(['// Warning: no "ParameterType" defined for', ns, '::', cls]))
                            continue
                    else:
                        print(' '.join(['// Warning: no parameter member defined for', ns, '::', mode]))
                        continue

                    # now we have a dict with all members of our parameter: params
                    # we need to inspect one by one
                    hlts = {}
                    jparams['params_opt'] = OrderedDict()
                    for p in parms:
                        tmp = splitns(p)[1]
                        if not tmp.startswith('_') and (ns not in ignore or tmp not in ignore[ns]):
                            hlt = self.to_hltype(ns, parms[p])
                            if hlt and hlt[1] in ['stdtype', 'enum', 'tm']:
                                (hlt, hlt_type, hlt_ns) = hlt
                                llt = splitns(parms[p])[1]
                                needed = True
                                pval = None
                                if hlt_type == 'enum':
                                    if len(self.namespace_dict[hlt_ns].enums[llt]) > 1:
                                        pval = '(' + hlt_ns + '::' + llt + ')string2enum_' + hlt_ns.replace(':', '_') + '[' + tmp + ']'
                                #elif llt == 'bool':
                                #    pval = 'string2bool(' + tmp + ')'
                                else:
                                    pval = tmp
                                if pval != None:
                                    thetype = (hlt if hlt else parms[p])
                                    if tmp in jparams['params_req']:
                                        td['params_req'][tmp] = pval
                                        decl_req.append('const ' + thetype + ' ' + tmp)
                                        call_req.append(tmp)
                                    else:
                                        td['params_opt'][tmp] = pval
                                        prm = tmp
                                        dflt = defaults[ns][prm] if ns in defaults and prm in defaults[ns] else self.defaults[thetype]
                                        decl_opt.append(' '.join(['const', thetype, prm, '=', dflt]))
                                        call_opt.append(prm)
                                else:
                                    print('// Warning: do not know what to do with ' + ns + ' ' + p)
                            else:
                                print('// Warning: parameter member ' + p + ' of ' + ns + ' is no stdtype, no enum and has no typemap. Ignored.')

            # endfor

            # Now let's get the input arguments (provided to input class/object of algos)
            iargs_decl = []
            iargs_call = []
            setinputs = ''
            inp = self.get_class_for_typedef(ns, 'Batch', 'InputType')
            if not inp and 'Input' in self.namespace_dict[ns].classes:
                inp = (ns, 'Input')
            if inp:
                expinputs = self.get_expand_attrs(inp[0], inp[1], 'sets')
                reqi = 0
                for ins, iname, itype in expinputs[0]:
                    tmpi = iname
                    if tmpi and (ns not in ignore or tmpi not in ignore[ns]):
                        if ns in defaults and tmpi in defaults[ns]:
                            i = len(iargs_decl)
                            dflt = ' = ' + defaults[ns][tmpi]
                        else:
                            i = reqi
                            reqi += 1
                            dflt = ''
                        if ns in has_dist and iname in has_dist[ns]['step_specs'][0].inputnames or iname in ['data', 'labels']:
                            itype = 'TableOrFList *'
                        iargs_decl.insert(i, 'const ' + itype + ' ' + iname + dflt)
                        iargs_call.insert(i, iname)
                        jparams['input_args'].insert(i, [ins + '::' + iname, iname, itype])
            else:
                print('// Warning: no input type found for ' + ns)


            jparams['iargs_decl'] = iargs_decl
            jparams['iargs_call'] = iargs_call
            jparams['decl_req'] = decl_req
            jparams['call_req'] = call_req
            jparams['decl_opt'] = decl_opt
            jparams['call_opt'] = call_opt
            # we will need something more sophisticated if the interesting parent class is not a direct parent (a grand-parent for example)
            prnts = list(set([cpp2hl(x) for x in ifaces if any(x.endswith(y) for y in self.namespace_dict[ns].classes[mode].parent)]))
            jparams['iface'] = prnts if len(prnts) else list([None])
        else:
            jparams = {}
            tdecl = []
        # here we know parameters, inputs etc for each
        # let's store this
        retjp = {
            'params': jparams,
            'sparams': tdecl,
            'model_typemap': self.prepare_modelmaps(ns),
            'result_typemap': self.prepare_resultmaps(ns),
        }
        if ns in has_dist:
            retjp['dist'] = has_dist[ns]

        return {ns + '::' + mode : retjp}


    def hlapi(self, lang, algo_patterns):
        """
        Generate high level wrappers for namespaces listed in algo_patterns (or all).

        First extract the namespaces we really want, e.g. ignore NN.

        Then we expand typedefs on class and namespace levels.

        Then generate maps for each algo mapping string arguments to C++ enum values.

        Next prepares parsed data for code generation (e.g. setting up dicst for jinja).

        Finally generates
          - type-converters, converting C++ types to native type (python-dict, R named list...)
          - algo-wrappers
          - initialization
        """
        tmaps, wrappers, hlapi, dtypes = '', '', '', ''
        algoconfig = {}
        
        algos = [x for x in self.namespace_dict if any(y in x for y in algo_patterns)] if algo_patterns else self.namespace_dict
        algos = [x for x in algos if not any(y in x for y in ['quality_metric', 'transform'])]

        # we first extract and prepare the data (input, parameters, results, template spec)
        # some algo need to combine several configs, like kmeans needs kmeans::init
        for ns in algos + ['algorithms::classifier', 'algorithms::linear_model',]:
            # expand typedefs
            self.expand_typedefs(ns)
            if not ns.startswith('algorithms::neural_networks'):
                if not any(ns.endswith(x) for x in ['objective_function', 'iterative_solver']):
                    tmp = ns.rsplit('::', 2)
                    if any(ns.endswith(x) for x in ['prediction', 'training', 'init', 'transform']):
                        func = '_'.join(tmp[1:])
                    else:
                        func = tmp[-1]
                    algoconfig.update(self.prepare_hlwrapper(ns, 'Batch', func))
        
        # and now we can finally generate the code
        wg = wrapper_gen(algoconfig, {cpp2hl(i): ifaces[i] for i in ifaces})
        cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end = '', '', '#define NO_IMPORT_ARRAY\n#include "daal4py_cpp.h"\n', '', '', ''
        
        for ns in algos:
            if ns.startswith('algorithms::') and not ns.startswith('algorithms::neural_networks') and self.namespace_dict[ns].enums:
                cpp_begin += 'static std::map< std::string, int > string2enum_' + ns.replace(':', '_') + ' =\n{\n'
                for e in  self.namespace_dict[ns].enums:
                    for v in self.namespace_dict[ns].enums[e]:
                        vv = ns + '::' + v
                        cpp_begin += ' '*4 +'{"' + v + '", ' + vv + '},\n'
                cpp_begin += '};\n\n'

        hlargs = {}
        for a in algoconfig:
            (ns, algo) = splitns(a)
            if algo.startswith('Batch'):
                tmp = wg.gen_wrapper(ns, algo)
                if tmp:
                    cpp_map   += tmp[0]
                    cpp_begin += tmp[1]
                    cpp_end   += tmp[2]
                    pyx_map   += tmp[3]
                    pyx_begin += tmp[4]
                    pyx_end   += tmp[5]
                    dtypes    += tmp[6]
                    hlargs[ns] = tmp[7]

        hds = wg.gen_headers()
        fts = wg.gen_footers(dtypes)

        pyx_end += fts[1]
        cpp_end += fts[0] + '\n/*\n'
        for algo in hlargs:
            if len(hlargs[algo]):
                cpp_end += 'Algorithm:' + cpp2hl(algo.replace('algorithms::', '')) + '\n'
                cpp_end += 'Name,Type,Default\n'
                for a in hlargs[algo]:
                    cpp_end += ','.join([str(x).rsplit('::')[-1] for x in a]).replace('const', '').replace('&', '').strip() + '\n'
                cpp_end += '\n'
        cpp_end += '\n*/\n'
        return(hds[0] + cpp_map + cpp_begin + '\n#endif', cpp_end, hds[1] + pyx_map + pyx_begin + pyx_end)

###############################################################################
###############################################################################
"""
FIXME
- template template and jinja macros need to work with all the data we extract
  - like generating imports and '#define USE_*' from deps
  - when producing the first versions for all the algorithms
    - think about simplifications, like converting all method values to lists
- almost everything we do manually in the template bodies should be moved
  to the cfg dicts, this allows much easier changes to all files.
  Moreover it will improve what we can do to compare C++ and our templates
- see which sections we currently do not compare and include in comparison
- some manual add_compute stuff can probably be simplified with the new tuple syntax
  for templates (see kmeans__init).
"""

###############################################################################
###############################################################################
###############################################################################
###############################################################################

description = """
A tool to create cython interface files for HLAPI of DAAL.
Extracting necessary data and creating internal data structures.
See parse.py for details about C++ parsing.
See wrappers.py for necessary configuration that can not be auto-extracted.
See wrapper_gen.py for code generation (cython and C++).
"""

def gen_daal4py(daalroot, outdir):
    iface = cython_interface(jp(daalroot, 'include', 'algorithms'))
    iface.read()
    cpp_h, cpp_cpp, pyx_file = iface.hlapi('', ['pca',
                                                'kmeans',
                                                'svd',
                                                'multinomial_naive_bayes',
                                                'linear_regression',
                                                'multivariate_outlier_detection',
                                                'univariate_outlier_detection',
                                                'svm',
                                                'kernel_function',
                                                'multi_class_classifier',
    ]
                                           # 'ridge_regression',
    )
    with open(jp(outdir, 'daal4py_cpp.h'), 'w') as f:
        f.write(cpp_h)
    with open(jp(outdir, 'daal4py_cpp.cpp'), 'w') as f:
        f.write(cpp_cpp)
    with open(jp(outdir, 'daal4py_cy.pyx'), 'w') as f:
        f.write(pyx_file)


if __name__ == "__main__":
    import argparse

    argParser = argparse.ArgumentParser(prog="gen_hlapi.py",
                                        description=description,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('--hlapi',    default=None, choices=['R', 'Python'], help="creates high level API for given language")
    argParser.add_argument('--daalroot', default=None,                          help="DAAL root directory (reads include dir in there)")
    argParser.add_argument('--outdir',   default='build',                       help="Output directory to store wrapper files to")
    

    args = argParser.parse_args()
    gen_daal4py(args.daalroot, args.outdir)
