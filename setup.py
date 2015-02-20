#!/usr/bin/env python

from setuptools import setup

setup(name="PerceptronixPointNever",
      version="1.1",
      description="Perceptronix Point Never, a POS tagger",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      url="http://github.com/cslu-nlp/PerceptronixPointNever/",
      install_requires=["nlup >= 0.3.0", "nltk >=3.0.0"],
      dependency_links=["http://github.com/cslu-nlp/nlup/archive/master.zip#egg=nlup-0.3.0"],
      packages=["perceptronixpointnever"])
