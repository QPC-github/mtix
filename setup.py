from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(name="mtix",
      version="2.0.0",
      description="MTIX MeSH main heading (Descriptor) and subheading (Qualifier) prediction pipeline.",
      long_description=readme(),
      url="https://github.com/ncbi/mtix/",
      author="Alastair Rae",
      author_email="",
      license="",
      packages=["mtix"],
      package_dir={"":"src"},
      python_requires=">=3.9",
      install_requires=[
          "pandas==1.4.1",
          "python-dateutil==2.8.2",
          "pytrec-eval==0.5",
          "sagemaker==2.80.0",
      ],
      include_package_data=True,
      zip_safe=False,
      )