#!/usr/bin/env python3 -O
# Copyright (C) 2014 Kyle Gorman
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

from __future__ import division

import logging

from sys import version_info

if version_info[0] != 3:
    logging.error('Requires Python 3')
    exit(1)

from string import digits
from itertools import product
from functools import partial
from collections import defaultdict
from numpy import int8, ones, zeros
from numpy.random import permutation
# timeit tests suggest that for generating a random list of indices of the
# sort used to randomize order of presentation, this is much faster than 
# `random.shuffle`; if for some reason you are unable to deploy `numpy`, 
# it should not be difficult to modify the code to use `random` instead.

## defaults and (pseudo)-globals
VERSION_NUMBER = 0.1
PRE_SUF_MAX = 4
TRAINING_ITERATIONS = 10
LEFT_PAD  = ['<S1>', '<S0>']
RIGHT_PAD = ['</S0>', '</S1>']

USAGE = """Perceptronix Point Never {}, by Kyle Gorman <gormanky@ohsu.edu>

USAGE: ./ppn.py INPUT_ARGS OUTPUT_ARG1 [...] [-h] [-v]

    Input arguments (exactly one required):

        -i tag         train model on "token/tag"-format data in `tagged`
        -p pickled     read pickled training model from `pickled`

    Output arguments (at least one required):

        -D pickled     dump pickled training model to `sink`
        -E tagged      report accuracy on already-tagged data in `tagged`
        -T untagged    tag data in `untagged`, writing to stdout
    
    Optional arguments:

        -t t           number of training iterations (default: {})
        -h             print this message and quit
        -v             increase verbosity
""".format(VERSION_NUMBER, TRAINING_ITERATIONS)

## generic helpers
# TODO do we need these?

def argmax(pairs):
    """
    Given a dictionary object, return the key whose value is maximum
    """
    return argmax_pair(pairs)[0]

def argmax_pair(pairs):
    """
    Given a dictionary object, return the key, value pair whose value is 
    maximum; ties resolved using dictionary order
    """
    return max(pairs.items(), key=lambda x: x[1])

## helper classes include a List-ifying decorator, a lazily-updated 
## averaged perceptron weight, and sequence tagger itself

class PoliteClass(object):
    """
    Abstract polite decorator
    """

    def __init__(self, function):
        self.function = function

    def __doc__(self):
        return self.function.__doc__

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
        return functools.partial(self.__call__, obj)

# container decorators

class Listify(PoliteClass):
    """
    Decorator which converts the output of a generator to a list

    >>> @Listify
    ... def fibonacci(n):
    ...     'Generator for the first n Fibonacci numbers'
    ...     F1 = 0
    ...     yield F1
    ...     F2 = 1
    ...     for _ in xrange(n - 1):
    ...         yield F2
    ...         (F1, F2) = (F2, F1 + F2)
    >>> print(fibonacci(10))
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """

    def __call__(self, *args, **kwargs):
        call = self.function(*args, **kwargs)
        return list(call) if call != None else []


class LazyWeight(object):
    """
    This class represents an individual weight in an averaged perceptron.
    It allows for several subtle optimizations. First, as the name 
    suggests, the `summed_weight` variable is lazily evaluated (i.e., 
    computed only when needed). This summed weight is the one used in 
    actual inference: we need not average explicitly. Lazy evaluation 
    requires us to store two other numbers. First, we store the "real" 
    weight (i.e., if this wasn't part of an averaged perceptron).  
    Secondly, we store the last time this weight was updated. These two 
    additional numbers work together as follows. When we need the real 
    value of the summed weight (for inference), we "evaluate" the summed 
    weight by adding to it the real weight multipled by the time elapsed.

    While passing around the time of the outer class is suboptimal, one
    advantage of this format is that we can store weights and their 
    times in the same place, reducing the number of redundant hash table 
    lookups required.

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

    def __init__(self, time, value=0):
        self.weight = value
        self.summed_weight = value
        self.timestamp = time

    def __repr__(self):
        return '{}(weight = {:d}, summed_weight = {:d}'.format(
                self.__class__.__name__, self.weight, self.summed_weight)

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


class PPN(object):
    """
    Perceptronix Point Never: an HMM tagger with fast discriminative 
    training using the perceptron algorithm
    """

    def __init__(self):
        self.time = 0
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
        logging.info('Constructed new PNN instance')

    # alternative constructor using pickled data

    @classmethod
    def from_model(cls, source):
        instance = cls()
        instance.load(source)
        return instance

    def train(self, corpus, T=1):
        logging.info('Extracting input features for training')
        # collect tagset and features 
        feature_corpus = []
        for sentence in corpus:
            (tokens, tags) = zip(*sentence)
            feature_corpus.append((tags, POS_token_features(tokens)))
            # "touch" each tag in weights to simplify Viterbi below
            for tag in tags:
                self.weights[tag]
        logging.info('Started {} epochs of training'.format(T))
        for t in range(T):
            for (tags, features) in permutation(feature_corpus):
                self._feature_tag(features)
                hypo_tags = self._feature_tag(features)
                gold_tags = tags
                # compare it to gold standard
                for (hypo_tag, gold_tag) in zip(hypo_tags, gold_tags):
                    # if it ain't broke, don't fix it
                    if hypo_tag == gold_tag:
                        continue
                    logging.debug('Hypothesized tag: ' + hypo_tag)
                    logging.debug('Gold tag: ' + gold_tag)
                    # reward and punish
                    self._update(gold_tag, features, +1)
                    self._update(hypo_tag, features, -1)
                    # increment time
                    self.time += 1
            logging.info('Epoch {} completed'.format(t + 1))
        logging.info('Completed {} epochs of training completed'.format(T))

    def _update(self, tag, features, sgn):
        """
        Apply update ("reward" if `sgn` == 1, "punish" if `sgn` == -1) for
        each feature in `features` for this `tag`
        """
        # lookup tag in weights dictionary
        tag_ptr = self.weights[tag]
        # update individual features
        for feat in features:
            tag_ptr[feat].update(self.time, sgn)

    def tag(tokens):
        """
        Tag a single `sentence` (list of tokens)
        """
        return self._feature_tag(POS_token_features(tokens))

    @Listify
    def batch_tag(self, corpus):
        """
        Tag a `corpus` (a list of `sentence`s, each a list of tokens)
        """
        for sentence in corpus:
            yield self.tag(sentence)

    def _feature_tag(self, features):
        """
        Using Viterbi decoding, tag a sentence, represented as a list of 
        sets containing precomputing input features of the corresponding 
        tokens
        """
        tagset = self.weights.keys()
        # initialize Viterbi trellis and backpointers
        size = ...
        trellis = zeros(size, dtype=int)
        backptrs = -ones(size, dtype=int8)  # are small natural numbers
        # extract tagset
        tagset = self.weights.keys()
        # populate it
        for (i, feature) in enumerate(features):
            for (tag, weights) in self.weights.items():
                emissions = sum(wval.get(self.time) for (wkey, wval) in \
                                weights.items() if wkey in features)
                for t_1 in tagset:
                    # do the max operation
                    bigram_weights = emissions + bigram_transitions
                    for t_2 in tagset:
                        # do the max operation
                        trigram_weights = bigram_weights + ...(t2, t1)
                        trellis[i][tag] = bigram_weights + ...(t2, t1)
                trellis.append(trellis_column)
        # follow backpointers
        # return tag_sequence
        return tag_sequence

    def evaluate(self, corpus):
        """
        Compute tag accuracy of the current model onto a `corpus`,
        represented as a list of `sentence`s, themselves lists of (`token`,
        `tag`) tuples
        """
        count = 0
        correct = 0
        for sentence in sentences:
            (tokens, gold_tags) = zip(*sentence)
            for (hypo_tag, gold_tag) in zip(self.tag(tokens), tags):
                count += 1
                correct += (hypo_tag == gold_tag)
        acc = correct / count
        logging.info('Evaluation completed: accuracy = {:4f}'.format(acc))
        return acc

## feature extractors

# features for POS tagging (from Ratnaparkhi 1996, but ignoring the 
# distinction between "rare" and "non-rare" words à la Collins 2002):
#
# w-2=X: two tokens back
# w-1=X: previous token
# w=X: current token
# w+1: next token
# w+2: two tokens ahead
# t-1=X: previous tag
# t-2,t-1=X: previous two tags
# p1=X: first character
# p2=X: first two characters
# p3=X: first three characters
# p4=X: first four characters
# s1=X: last character
# s2=X: last two characters
# s3=X: last three characters
# s4=X: last four characters
# h: contains a hyphen?
# n: contains a number?
# u: contains an uppercase character

@Listify
def POS_token_features(tokens):
    """
    Given lists of tokens, extract all token-related features in the form
    of a list of sets where each set contains the (non-zero) token-related
    features for the corresponding token

    The rules for are a bit complex, but that's how the standard reference 
    (Ratnaparkhi 1997) did it.
    """
    folded_tokens = [t.lower() for t in tokens]
    padded_tokens = LEFT_PAD + folded_tokens + RIGHT_PAD
    for (i, ftoken) in enumerate(padded_tokens[2:-2]):
        featset = {'b'}  # initialize with bias term
        # tokens nearby
        featset.add('w-2={}'.format(padded_tokens[i]))
        featset.add('w-1={}'.format(padded_tokens[i + 1]))
        featset.add('w={}'.format(ftoken))  # == padded_tokens[i + 2]
        featset.add('w+1={}'.format(padded_tokens[i + 3]))
        featset.add('w+2={}'.format(padded_tokens[i + 4]))
        # "prefix" and "suffix" features
        for j in range(1, 1 + min(len(ftoken), PRE_SUF_MAX)):
            featset.add('p{}={}'.format(j, ftoken[:+j]))  # prefix
            featset.add('s{}={}'.format(j, ftoken[-j:]))  # suffix
        # contains a hyphen?
        if any(c == '-' for c in ftoken):
            featset.add('h')
        # contains a number?
        if any(c in digits for c in ftoken):
            featset.add('n')
        # contains an uppercase character?
        if ftoken != tokens[i]:
            featset.add('u')
        # and we're done with that word
        yield featset

@Listify
def POS_tag_features(tags):
    """
    Given lists of tokens, extract trigram tag-related features in the 
    form of a list of sets (as above)
    """
    padded_tags = LEFT_PAD + list(tags)
    for i in range(len(padded_tags) - 2):
        yield {'t-1={}'.format(padded_tags[i + 1]),
               't-2,t-1={},{}'.format(*padded_tags[i:i + 2])}

# TODO NP-chunking features (from Collins 2002):
#
# def NPC_token_features(tokens):
# def NPC_tag_features(tags):
#
# b: bias (omnipresent)
#
# w-1=X: previous token
# w-2=X: two tokens back
# w=X: current token
# w+1=X: next token
# w+2=X: two tokens ahead
#
# w-2,w-1=X,Y: previous two tokens
# w-1,w=X,Y: previous token and current token
# w,w+1=X,Y: current token and next token
# w+1,w+2=X,Y: next two tokens
#
# t-1=X: previous tag
# t-2=X: two tags back
# t=X: current tag
# t+1=X: next tag
# t+2=X: two tags ahead
# 
# t-2,t-1=X,Y: previous two tags
# t-1,t=X,Y: previous tag and current tag
# t,t+1=X,Y: next two tags
# 
# t-2,t-1,t=X,Y,Z: previous two tags and current tag
# t-1,t,t+1=X,Y,Z: previous tag, current tag, and next tag
# t,t+1,t+2=X,Y,Z: current tag and next two tags


if __name__ == '__main__':

    from gzip import GzipFile
    from sys import argv, stderr
    from pickle import dump, load
    from nltk import str2tuple, untag
    from getopt import getopt, GetoptError

    ## helpers

    @Listify
    def tag_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wt) for wt in line.strip().split()]

    @Listify
    def untag_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield untag([str2tuple(wt) for wt in line.strip().split()])

    ## parse arguments

    try:
        (optlist, args) = getopt(argv[1:], 'i:p:D:E:T:t:hv')
    except GetoptError as err:
        logging.error(err)
        exit(USAGE)
    # warn users about arguments not from opts (as this is unsupported)
    for arg in args:
        logging.warning('Ignoring command-line argument "{}"'.format(arg))
    # set defaults
    test_source = None
    tagged_source = None
    untagged_source = None
    model_source = None
    model_sink = None
    training_iterations = TRAINING_ITERATIONS
    # read optlist
    for (opt, arg) in optlist:
        if opt == '-i':
            tagged_source = arg
        elif opt == '-p':
            model_source = arg
        elif opt == '-D':
            model_sink = arg
        elif opt == '-E':
            test_source = arg
        elif opt == '-T':
            untagged_source = arg
        elif opt == '-t':
            try:
                training_iterations = int(arg)
            except ValueError:
                logging.error('Cannot parse -t arg "{}"'.format(arg))
                exit(USAGE)
        elif opt == '-h':
            exit(USAGE)
        elif opt == '-v':
            logging.basicConfig(level=logging.INFO)
        else:
            logging.error('option {} not found'.format(opt, arg))
            exit(USAGE)
    
    ## run it

    # model input
    ppn = None
    if tagged_source:
        if model_source:
            logging.error('Incompatible inputs (-i and -p) specified')
            exit(USAGE)
        logging.info('Training model with tagged data "{}"'.format(
                                                            tagged_source))
        ppn = PPN()
        ppn.train(tag_reader(tagged_source), training_iterations)
    elif model_source:
        logging.info('Initializing model from pickled data "{}"'.format(
                                                           model_source))
        try:
            with GzipFile(model_source) as source:
                ppn = PPN.from_model(source)
        except (IOError, FileNotFoundError) as err:
            logging.error(err)
            exit(USAGE)
    else:
        logging.error('No input specified')
        exit(USAGE)
    # model outputs
    if not any(model_sink, untagged_source, tagged_source):
        logging.error('No outputs specified')
        exit(USAGE)
    if model_sink:
        logging.info('Writing pickled data to "{}'.format(model_sink))
        with GzipFile(model_sink, 'w') as sink:
            ppn.dump(sink)
    if untagged_source:
        logging.info('Tagging untagged data from "{}"'.format(
                                              untagged_source))
        with open(untagged_source) as source:
            for line in ppn.batch_tag(source):
                print(' '.join(tuple2str(w, t) for (w, t) in line))
    if tagged_source:
        logging.info('Performing evaluation on tagged data "{}"'.format(
                                                            test_source))
        with open(untagged_source) as source:
            accuracy = ppn.evaluate(tag_reader(untagged_source))
            print('Accuracy: {:4f}'.format(accuracy))
