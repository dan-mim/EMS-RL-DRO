import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="pyreductree",
      version="1.0",
      author="D. Mimouni, P.Malisani",
      author_email="daniel.mimouni@ifpen.fr",
      description="Scenario reduction package",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['pyreductree'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent"],
      include_package_data=True,
      install_requires=[
          "numpy>=1.26.4",
          "scipy>=1.11.4",
          "casadi>=3.6.5",
          "pyopticontrol>=3.0",
          "pystocoptim>=1.0"
      ],
)