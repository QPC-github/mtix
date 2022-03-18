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
      install_requires=[],
      include_package_data=True,
      zip_safe=False)