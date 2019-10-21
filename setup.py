from setuptools import setup

setup(name='gmlsnets-tensorflow',
      version='0.1',
      description='GMLS-Nets Tensorflow implementation',
      url='http://github.com/rgp62/gmls-nets',
      author='Ravi G. Patel and Nathaniel Trask',
      author_email='rgpatel@sandia.gov',
      packages=['gmlsnets_tensorflow'],
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-learn',
          'toolz',
          'tensorflow',
        ],
)
