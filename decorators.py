#!/usr/bin/env python -O
# Copyright (C) 2014 Kyle Gorman & Stephen Bedrick
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
# decorators.py: decorators

from __future__ import division

from numpy import array
from functools import partial

# "metadecorators"


class PoliteClass(object):

    """
    Abstract polite decorator
    """

    def __init__(self, function):
        self.function = function
        self.__doc__ = self.function.__doc__
        self.__name__ = self.function.__name__

    def __repr__(self):
        return repr(self.function)

    def __str__(self):
        return str(self.function)

    def __name__(self):
        return self.function.__name__

    def __call__(self, *args, **kwargs):
        """
        Default, do-nothing definition
        """
        return self.function(*args, **kwargs)

    def __get__(self, obj):
        """
        Access instance methods
        """
        return partial(self.__call__, obj)


# container decorators

class Listify(PoliteClass):

    """
    Decorator which converts the output of a generator (or whatever) to a
    list

    >>> @Listify
    ... def fibonacci(n):
    ...     'Generator for the first n Fibonacci numbers'
    ...     F1 = 0
    ...     yield F1
    ...     F2 = 1
    ...     for _ in xrange(n - 1):
    ...         yield F2
    ...         (F1, F2) = (F2, F1 + F2)
    >>> print fibonacci(10)
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    >>> print fibonacci.__doc__
    Generator for the first n Fibonacci numbers
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return list(call) if call != None else list()


class Setify(PoliteClass):

    """
    Decorator which converts the output of a generator (or whatever) to a
    set (hash-backed container for unique elements)
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return set(call) if call is not None else set()


class Tupleify(PoliteClass):

    """
    Decorator which converts the output of a generator (or whatever) to a
    tuple
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return tuple(call) if call is not None else tuple()


class Arrayify(PoliteClass):

    """
    Decorator which converts the output of a generator (or whatever) to
    a numpy array
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return array(list(call)) if call is not None else array()


# sorting

class Sortify(PoliteClass):

    """
    Decorator which sorts the output of a generator(like) function

    >>> @Sortify
    ... def dumb():
    ...     yield 3
    ...     yield 1
    ...     yield 4
    ...     yield 5
    >>> dumb()
    [1, 3, 4, 5]
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return sorted(call) if call is not None else list()


# apply zip(*retval)

class Zipstarify(PoliteClass):
    
    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return zip(*call) if call is not None else None


# memoization

class Memoize(PoliteClass):

    """
    Decorator which caches a function's return value each time it is
    called; if called later with the same arguments, the cached value is
    returned rather than re-evaluating the function

    The following test confirms that memoization works (if this test takes
    more than 100ms, memoization is broken) and that it is "polite" in
    the sense that it exposes __doc__ of the underlying function.

    >>> @Memoize
    ... def fibonacci(n):
    ...     'A recursive Fibonacci number function that actually works'
    ...     if n in (0, 1):
    ...         return n
    ...     return fibonacci(n - 1) + fibonacci(n - 2)
    >>> print fibonacci(100)
    354224848179261915075
    >>> print fibonacci.__doc__
    A recursive Fibonacci number function that actually works
    """

    def __init__(self, function):
        self.cache = {}
        self.function = function
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__

    def __call__(self, *args, **kwargs):
        if not hasattr(args, '__hash__'):  # uncacheable
            return self.function(*args, **kwargs)
        if args in self.cache:             # cached
            return self.cache[args]
        else:                              # soon to be cached
            value = self.function(*args, **kwargs)
            self.cache[args] = value
            return value


if __name__ == '__main__':
    import doctest
    doctest.testmod()
