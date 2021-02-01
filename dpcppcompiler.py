import distutils.ccompiler
import sys

_compiler_module, _compiler_class, _ = distutils.ccompiler.compiler_class["msvc"]

_msvc_compiler_module_name = "distutils." + _compiler_module
__import__(_msvc_compiler_module_name)
_msvc_compiler_module = sys.modules[_msvc_compiler_module_name]
cls = vars(_msvc_compiler_module)[_compiler_class]


class DPCPPCompiler(cls):
    def initialize(self):
        super().initialize()
        self.cc = "dpcpp.exe"
