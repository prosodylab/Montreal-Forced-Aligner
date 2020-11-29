import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


def readme():
    with open('README.md') as f:
        return f.read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        if __name__ == '__main__':  # Fix for multiprocessing infinite recursion on Windows
            import pytest
            errcode = pytest.main(self.test_args)
            sys.exit(errcode)


if __name__ == '__main__':
    setup(name='Montreal Forced Aligner',
          version='2.0.0a',
          description='',
          long_description='',
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Operating System :: OS Independent',
              'Topic :: Scientific/Engineering',
              'Topic :: Text Processing :: Linguistic',
          ],
          keywords='phonology corpus phonetics alignment segmentation',
          url='https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner',
          author='Montreal Corpus Tools',
          author_email='michael.e.mcauliffe@gmail.com',
          packages=['montreal_forced_aligner',
                    'montreal_forced_aligner.aligner',
                    'montreal_forced_aligner.g2p',
                    'montreal_forced_aligner.command_line',
                    'montreal_forced_aligner.config',
                    'montreal_forced_aligner.features',
                    'montreal_forced_aligner.trainers',
                    'montreal_forced_aligner.gui',
                    'montreal_forced_aligner.lm'],
          install_requires=[
              'textgrid',
              'tqdm',
              'alignment',
              'requests',
              'pyyaml',
              'librosa',
              'pyqt5',
              'pyqtgraph'
          ],
          entry_points={
              'console_scripts': ['mfa=montreal_forced_aligner.command_line.mfa:main']
          },
          package_data={'montreal_forced_aligner.config': ['*.yaml']},
          cmdclass={'test': PyTest},
          extras_require={
              'testing': ['pytest'],
          }
          )
