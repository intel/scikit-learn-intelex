from distutils.msvccompiler import MSVCCompiler


class DPCPPCompiler(MSVCCompiler):
    def initialize(self):
        super().initialize()
        self.cc = "dpcpp.exe"
