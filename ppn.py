#!/usr/bin/env python -O
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
#
# Perceptronix Point Never: a perceptron-based part-of-speech tagger

from __future__ import division

import logging

from time import time
from collections import defaultdict
#from numpy import int8, ones, zeros

from numpy.random import permutation
# timeit tests suggest that for generating a random list of indices of the
# sort used to randomize order of presentation, this is much faster than
# `random.shuffle`; if for some reason you are unable to deploy `numpy`,
# it should not be difficult to modify the code to use `random` instead.
from jsonpickle import encode, decode

from lazyweight import LazyWeight
from features import POS_token_features

## defaults and (pseudo)-globals
VERSION_NUMBER = 0.2
TRAINING_ITERATIONS = 10

## usage string
USAGE = """Perceptronix Point Never {}, by Kyle Gorman <gormanky@ohsu.edu>

    Input arguments (exactly one required):

        -i tag         train model on data in `tagged`
        -p source      read serialized model from `source`

    Output arguments (at least one required):

        -D sink        dump serialized training model to `sink`
        -E tagged      compute accuracy on data in `tagged`
        -T untagged    tag data in `untagged`
    
    Optional arguments:

        -t t           number of training iterations (default: {})
        -h             print this message and quit
        -v             increase verbosity

Options `-i` and `-E` take whitespace-delimited "token/tag" pairs as input.
Option `-T` takes whitespace-delimited tokens (no tags) as input.
""".format(VERSION_NUMBER, TRAINING_ITERATIONS)


class PPN(object):

    """
    Perceptronix Point Never: an HMM tagger with fast discriminative 
    training using the perceptron algorithm
    """

    def __init__(self, sentences=None, T=1):
        self.time = 0
        # the (outer) keys are features represented as strings; the values
        # are (inner) dictionaries with tag keys and LazyWeight values
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
        logging.info('Constructed new PPN instance.')
        if sentences:
            self.train(sentences, T)

    # alternative constructor using JSON
    
    @classmethod
    def load(cls, source):
        """
        Create new PPN instance from serialized JSON from `source`
        """
        return decode(source.read())

    def dump(self, sink):
        """
        Serialize object (as JSON) and print to `sink`
        """
        print >> sink, encode(self)

    def train(self, sentences, T=1):
        logging.info('Extracting input features for training.')
        # collect tagset and features
        self.tagset = set()
        feature_corpus = []
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            feature_corpus.append((tags, POS_token_features(tokens)))
            self.tagset.update(tags)
        # begin training
        logging.info('{} epoch(s) of training...'.format(T))
        for t in range(T):
            tic = time()
            epoch_right = 0
            epoch_wrong = 0
            for (gtags, features) in permutation(feature_corpus):
                # get hypothesized tagging
                htags = self._feature_tag(features)
                # compare it to gold standard
                for (htag, gtag, featset) in zip(htags, gtags, features):
                    # if it ain't broke, don't fix it
                    if htag == gtag:
                        epoch_right += 1
                        continue
                    logging.debug('Htag {} -> Gtag {}'.format(htag, gtag))
                    # reward and punish
                    self._update(gtag, featset, +1)
                    self._update(htag, featset, -1)
                    epoch_wrong += 1
                # increment time
                self.time += 1
            # duration of the epoch in sec
            # compute accuracy
            acc = epoch_right / (epoch_right + epoch_wrong)
            if acc == 1.:
                logging.warning('Early convergence! Woohoo!')
            logging.info('Epoch {:02} acc.: {:.04f}'.format(t + 1, acc) +
                         ' ({}s elapsed).'.format(int(time() - tic)))
        logging.info('Training complete.')

    def _update(self, tag, featset, sgn):
        """
        Apply update ("reward" if `sgn` == 1, "punish" if `sgn` == -1) for
        each feature in `features` for this `tag`
        """
        # update features for this tag
        tag_ptr = self.weights[tag]
        for feat in featset:
            tag_ptr[feat].update(self.time, sgn)

    def tag(self, tokens):
        """
        Tag a single `sentence` (list of tokens)
        """
        return zip(tokens, self._feature_tag(POS_token_features(tokens)))

    def _feature_tag(self, features):
        """
        Using Viterbi decoding, tag a sentence, represented as a list of 
        sets containing precomputing input features of the corresponding 
        tokens; note this does not return (token, tag) tuples but just 
        a tag sequence
        """
        tags = []
        for featset in features:
            best_tag = None
            best_score = None  # None is less than every number
            for tag in self.tagset:
                tag_ptr = self.weights[tag]
                tag_score = sum(tag_ptr[feat].get(self.time) for
                                feat in featset)
                if tag_score > best_score:
                    best_tag = tag
                    best_score = tag_score
            tags.append(best_tag)
        return tags
        # FIXME not actually using Viterbi decoding yet
        """
        summed_weights = defaultdict(int)
        featweight_ptr = self.weights[feat]
        for (tag, weight) in featweight_ptr.iteritems():
            summed_weights[tag] += weight
        # initialize Viterbi trellis and backpointers
        #size = ...
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
                    #for t_2 in tagset:
                        # do the max operation
                        #trigram_weights = bigram_weights + ...(t2, t1)
                        #trellis[i][tag] = bigram_weights + ...(t2, t1)
                trellis.append(trellis_column)
        # follow backpointers
        # return tag_sequence
        return tag_sequence
        """

    def evaluate(self, sentences):
        """
        Compute tag accuracy of the current model using a held-out list of
        `sentence`s (list of token/tag pairs)
        """
        total = 0
        correct = 0
        for sentence in sentences:
            (tokens, gtags) = zip(*sentence)
            htags = [tag for (token, tag) in self.tag(tokens)]
            for (htag, gtag) in zip(htags, gtags):
                total += 1
                correct += (htag == gtag)
        return correct / total


if __name__ == '__main__':

    from sys import argv
    from gzip import GzipFile
    from nltk import str2tuple, untag
    from getopt import getopt, GetoptError

    from decorators import Listify

    # helpers

    @Listify
    def tag_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wt) for wt in line.strip().split()]

    @Listify
    def untagged_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield line.strip().split()

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
                logging.error('Cannot parse -t arg "{}".'.format(arg))
                exit(USAGE)
        elif opt == '-h':
            exit(USAGE)
        elif opt == '-v':
            logging.basicConfig(level=logging.INFO)
        else:
            logging.error('Option {} not found.'.format(opt, arg))
            exit(USAGE)

    ## check outputs
    if not any((model_sink, untagged_source, test_source)):
        logging.error('No outputs specified.')
        exit(USAGE)

    ## run inputs
    ppn = None
    if tagged_source:
        if model_source:
            logging.error('Incompatible inputs (-i and -p) specified.')
            exit(1)
        logging.info('Training model from tagged data "{}".'.format(
                                                            tagged_source))
        try:
            ppn = PPN(tag_reader(tagged_source), training_iterations)
        except IOError as err:
            logging.error(err)
            exit(1)
    elif model_source:
        logging.info('Reading model from serialized data "{}".'.format(
                                                             model_source))
        try:
            with GzipFile(model_source, 'r') as source:
                ppn = PPN.load(source)
        except IOError as err:
            logging.error(err)
            exit(1)
    else:
        logging.error('No input specified.')
        exit(USAGE)

    ## run outputs
    if test_source:
        logging.info('Evaluating on data from "{}".'.format(test_source))
        try:
            with open(test_source, 'r') as source:
                accuracy = ppn.evaluate(tag_reader(test_source))
                print 'Accuracy: {:4f}'.format(accuracy)
        except IOError as err:
            logging.error(err)
            exit(1)
    if untagged_source:
        logging.info('Tagging data from "{}".'.format(untagged_source))
        try:
            for tokens in untagged_reader(untagged_source):
                print ' '.join(tuple2str(token, tag) for
                               (token, tag) in ppn.tag(tokens))
        except IOError as err:
            logging.error(err)
            exit(1)
    if model_sink:
        logging.info('Writing serialized data to "{}.'.format(model_sink))
        try:
            with GzipFile(model_sink, 'w') as sink:
                ppn.dump(sink)
        except IOError as err:
            logging.error(err)
            exit(1)
