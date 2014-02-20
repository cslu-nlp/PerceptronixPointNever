#!/usr/bin/env python -O
# Copyright (C) 2014 Kyle Gorman & Steven Bedrick
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Perceptronix Point Never: a perceptron-based part-of-speech tagger
# lazyweight.py: a lazily-evaluated averaged perceptron weight


from __future__ import division

class LazyWeight(object):

    """
    This class represents an individual weight in an averaged perceptron
    as a signed integer. It allows for several subtle optimizations. 
    First, as the name suggests, the `summed_weight` variable is lazily 
    evaluated (i.e., computed only when needed). This summed weight is the
    one used in actual inference: we need not average explicitly. Lazy 
    evaluation requires us to store two other numbers. First, we store the
    "real" weight (i.e., if this wasn't part of an averaged perceptron).
    Secondly, we store the last time this weight was updated. These two 
    additional numbers work together as follows. When we need the real 
    value of the summed weight (for inference), we "evaluate" the summed 
    weight by adding to it the real weight multipled by the time elapsed.

    While passing around the time of the outer class is suboptimal, one
    advantage of this format is that we can store weights and their times 
    in the same place, reducing the number of redundant hashtable lookups 
    required.

    # initialize 
    >>> t = 0
    >>> lw = LazyWeight(t, 1)
    >>> lw.get(t)
    1

    # some time passes...
    >>> t += 1
    >>> lw.get(t)
    2

    # weight is now changed
    >>> lw.update(t, -1)
    >>> t += 3
    >>> lw.update(t, -1)
    >>> t += 3
    >>> lw.get(t)
    -1
    """

    def __init__(self, time=0, weight=0):
        self.weight = weight
        self.summed_weight = weight
        self.timestamp = time

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def _evaluate(self, time):
        """
        This function applies queued updates and freshens the timestamp, 
        and, should be called any time the value of a weight is used or
        modified
        """
        if time == self.timestamp:
            return
        self.summed_weight += (time - self.timestamp) * self.weight
        self.timestamp = time

    def get(self, time):
        """
        Return an up-to-date sum of weights
        """
        self._evaluate(time)
        return self.summed_weight

    def update(self, time, value):
        """
        Bring sum of weights up to date, then add `value` to the weight
        """
        self._evaluate(time)
        self.weight += value
