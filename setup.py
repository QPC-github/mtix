from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mtix_descriptor_prediction_pipeline',
      version='0.1',
      description='MTIX MeSH Descriptor prediction pipeline.',
      long_description=readme(),
      url='https://github.com/ncbi/mtix/',
      author='Alastair Rae',
      author_email='',
      license='',
      packages=['mtix_descriptor_prediction_pipeline'],
      python_requires='3.10.2'
      install_requires=[
          "pandas==1.4.1",
          "python-dateutil==2.8.2",
          "pytrec-eval==0.5",
          "sagemaker-python-sdk==2.80.0",
          "zlib==1.2.11"
      ],
      include_package_data=True,
      zip_safe=False
      test_suite='nose.collector',
      tests_require=['nose']
      )