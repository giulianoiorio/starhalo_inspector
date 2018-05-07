#from setuptools import setup
import shutil
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import stat



cy=['pycam/cutils.pyx']
cy_ext=Extension('pycam/cutils',sources=cy)

ext_modules=cythonize(cy_ext)

setup(
		name='pycam',
		version='2.7.2',
		author='Giuliano Iorio',
		author_email='',
		url='',
		packages=['pycam'],
        scripts=['pycam/script/makehalo'],
        ext_modules=ext_modules
	)


	

try:
    shutil.rmtree('build')
    shutil.rmtree('dist')
    shutil.rmtree('pycam.egg-info')
    print('Done')
except:
    print('Done')
