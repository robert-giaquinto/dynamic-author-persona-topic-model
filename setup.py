from __future__ import absolute_import
import os
import sys
import io
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

if sys.version_info[:2] < (2, 7):
    raise Exception('This version of gensim needs Python 2.7 or later.')


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


class Install(_install):
    def run(self):
        _install.do_egg_install(self)


setup(name='Dynamic Author-Person Topic Model',
      version='0.1',
      description='Base Python implementation of the Dynamic Author-Persona Topic Model.',
      long_description=readfile('README.md'),
      url='https://github.com/robert-giaquinto/dynamic-author-persona-topic-model',
      author='Robert Giaquinto',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      cmdclass={'install': Install},
      install_requires=['numpy', 'scipy'],
      setup_requires=['numpy', 'scipy'],
      zip_safe=False)
