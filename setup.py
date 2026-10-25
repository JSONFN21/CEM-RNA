# build the fast sequence converter and lazyoutside scorer
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

Path('build').mkdir(exist_ok=True)

extensions = [Pybind11Extension('cem_rna._lazyoutside_native', ['src/native/lazyoutside_binding.cpp'], define_macros=[('lpv', '1')], cxx_std=17)]
setup(ext_modules=extensions, cmdclass={'build_ext': build_ext}, options={'egg_info': {'egg_base': 'build'}})
