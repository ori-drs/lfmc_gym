import os
import sys
import subprocess
import platform

from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.core import setup

__CMAKE_PREFIX_PATH__ = None
__DEBUG__ = False

if "--CMAKE_PREFIX_PATH" in sys.argv:
    index = sys.argv.index('--CMAKE_PREFIX_PATH')
    __CMAKE_PREFIX_PATH__ = sys.argv[index + 1]
    sys.argv.remove("--CMAKE_PREFIX_PATH")
    sys.argv.remove(__CMAKE_PREFIX_PATH__)

if "--Debug" in sys.argv:
    index = sys.argv.index('--Debug')
    sys.argv.remove("--Debug")
    __DEBUG__ = True


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        self.build_temp = 'build'

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)) + '/raisim_gym_torch/env/bin')
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        if __CMAKE_PREFIX_PATH__ is not None:
            cmake_args.append('-DCMAKE_PREFIX_PATH=' + __CMAKE_PREFIX_PATH__)

        cfg = 'Debug' if __DEBUG__ else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='lfmc_gym',
    version='0.2',
    author='Siddhant Gangapurwala',
    packages=find_packages(),
    author_email='siddhant@gangapurwala.com',
    description='Training repository for low-frequency motion control policies.',
    long_description='',
    ext_modules=[CMakeExtension('_lfmc_gym')],
    install_requires=['ruamel.yaml', 'numpy', 'torch', 'tensorboard>=1.15', 'pillow', 'psutil'],
    cmdclass=dict(build_ext=CMakeBuild),
    include_package_data=True,
    zip_safe=False,
)
