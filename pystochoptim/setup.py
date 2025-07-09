import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="pystocoptim",
      version="1.0",
      author="Malisani P.",
      author_email="paul.malisani@ifpen.fr",
      description="stochastic optimal control solver package",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['pystocoptim'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent"],
      include_package_data=True,
      install_requires=[
          "numpy>=1.26.4",
          "scipy>=1.11.4",
          "casadi>=3.6.5",
          "pyopticontrol>=3.0"
      ],
)