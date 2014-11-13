#!/usr/bin/env python

from setuptools import setup

setup(name="ppn",
      version="1.0",
      description="Perceptronix Point Never, a POS tagger",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      url="http://github.com/cslu-nlp/PerceptronixPointNever/",
      install_requires=["nlup >= 0.2.0", "nltk >=3.0.0"],
      dependency_links=["http://github.com/cslu-nlp/nlup/archive/master.zip#egg=nlup-0.2.0"],
      packages=["ppn"])
