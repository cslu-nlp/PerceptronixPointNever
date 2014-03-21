#!/usr/bin/env python -O
# encoding: UTF-8
#
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
#
# TODO:
#   * add IOB chunking support

from __future__ import division

import logging
import jsonpickle

from time import time
from collections import defaultdict

from numpy.random import permutation
# timeit tests suggest that for randomizing order of presentation,
# `permutation` is much faster than `random.shuffle`; if for some reason
# you are unable to use `numpy`, it should not be at all difficult to
# modify the code to use `random` instead.
from numpy import arange, copy, uint16, unravel_index, zeros

from lazyweight import LazyWeight
from decorators import Listify, Zipstarify
from features import bigram_tf, trigram_tf, extract_sent_efs, \
    extract_sent_tfs

## defaults and (pseudo)-globals
VERSION_NUMBER = 0.7
TRAINING_ITERATIONS = 10

# usage string
USAGE = """Perceptronix Point Never {0}, by Kyle Gorman and Steven Bedrick

    {1} [-i|-p input] [-D|-E|-T output] [-t {2}] [-h] [-v]

    Input arguments (exactly one required):

        -i tag         train model on data in `tagged`
        -p source      read serialized model from `source`

    Output arguments (at least one required):

        -D sink        dump serialized training model to `sink`
        -E tagged      compute accuracy on data in `tagged`
        -T untagged    tag data in `untagged`
    
    Optional arguments:

        -t t           number of training iterations (-i only) [{2}]
        -h             print this message and quit
        -v             increase verbosity

Options `-i` and `-E` take whitespace-delimited "token/tag" pairs as input.
Option `-T` takes whitespace-delimited tokens (no tags) as input.
""".format(VERSION_NUMBER, __file__, TRAINING_ITERATIONS)

# helpers


class PPN(object):

    """
    Perceptronix Point Never: an HMM tagger with fast discriminative
    training using the perceptron algorithm

    This implements the `nltk.tag.TaggerI` interface.
    """

    def __repr__(self):
        return '{}(time = {})'.format(self.__class__.__name__, self.time)

    def __init__(self, sentences=None, T=1):
        """
        Initialize the model, if a list(list(str, str)) of `sentences` is
        provided, perform `T` epochs of training
        """
        self.time = 0
        # structure of the weights:
        #
        # * outer keys are feature strings
        # * inter keys are tag strings
        # * values are LazyWeights
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
        self.tagset = set()
        logging.info('Constructed new PPN instance.')
        if sentences:
            self.train(sentences, T)

    # alternative constructor using serialized JSON

    @classmethod
    def load(cls, source):
        """
        Create new PPN instance from serialized JSON from `source`
        """
        retval = jsonpickle.decode(source.read(), keys=True)
        retval._update_tagset_cache()
        retval._update_transition_cache()
        return retval

    def dump(self, sink):
        """
        Serialize object (as JSON) and write to `sink`
        """
        print >> sink, jsonpickle.encode(self, keys=True)

    @staticmethod
    @Zipstarify
    def corpus_fs(sentences):
        """
        Extract all features used in training, returning a (tag list,
        token feature list, and tag feature list) tuple for each sentence;
        """
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            yield (tags, extract_sent_efs(tokens), extract_sent_tfs(tags))

    def _update_tagset_cache(self, corpus_tags=None):
        """
        Update the cache of tagset information (and also the transition
        cache, which uses the relevant indices)
        """
        # tagset
        if corpus_tags:
            for tags in corpus_tags:
                self.tagset.update(tags)
        self.Lt = len(self.tagset)
        # mapp between tag, index, and string
        self.idx2tag = {i: tag for (i, tag) in enumerate(self.tagset)}
        self.tag2idx = {tag: i for (i, tag) in self.idx2tag.iteritems()}
        self.idx2bigram_tfs = {i: bigram_tf(prev_tag) for (i, prev_tag)
                               in self.idx2tag.iteritems()}

    def _update_transition_cache(self):
        """
        Update the cache of bigram transition weights
        """
        self.btf_weights = zeros((self.Lt, self.Lt), dtype=int)
        # tags at t are the first axis, tags at t - 1 are the second
        for (i, bigram_tfs) in self.idx2bigram_tfs.iteritems():
            col = self.btf_weights[:, i]
            for (tag, weight) in self.weights[bigram_tfs].iteritems():
                col[self.tag2idx[tag]] += weight.get(self.time)

    def train(self, sentences, T=1):
        """
        Perform `T` epochs of training using data from `sentences`, a
        list(list(str, str)), as generated by `__main__.tag_reader`
        """
        logging.info('Extracting input features for training.')
        (corpus_tags, corpus_efs, corpus_tfs) = PPN.corpus_fs(sentences)
        epoch_size = sum(len(sent_tags) for sent_tags in corpus_tags)
        self._update_tagset_cache(corpus_tags)
        # begin training
        logging.info('Beginning {} epochs of training...'.format(T))
        for t in xrange(1, T + 1):
            tic = time()
            epoch_wrong = 0
            for (g_tags, sent_efs, sent_tfs) in permutation(zip(
                    corpus_tags, corpus_efs, corpus_tfs)):
                # generate hypothesized tagging and compare to gold tagging
                self._update_transition_cache()
                h_tags = self._feature_tag(sent_efs)
                for (h_tag, g_tag, token_efs, token_tfs) in zip(h_tags,
                            g_tags, sent_efs, sent_tfs):
                    if h_tag == g_tag:
                        continue
                    self._update(token_efs + token_tfs, g_tag, h_tag)
                    epoch_wrong += 1
                self.time += 1
            # compute accuracy
            accuracy = 1. - (epoch_wrong / epoch_size)
            logging.info('Epoch {:02} acc.: {:.04f}'.format(t, accuracy) +
                         ' ({}s elapsed).'.format(int(time() - tic)))
        self._update_transition_cache()
        logging.info('Training complete.')

    def _update(self, token_fs, g_tag, h_tag):
        """
        Update weights, rewarding the correct tag `g_tag` and penalizing
        the incorrect tag `h_tag`
        """
        for feat in token_fs:
            featptr = self.weights[feat]
            featptr[g_tag].update(+1, self.time)
            featptr[h_tag].update(-1, self.time)

    def _e_weights(self, token_efs):
        """
        Use a vector of token features (representing a single state) to
        compute emission weights for a state with `token_efs` features
        """
        e_weights = zeros(self.Lt, dtype=int)
        for feat in token_efs:
            for (tag, weight) in self.weights[feat].iteritems():
                e_weights[self.tag2idx[tag]] += weight.get(self.time)
        return e_weights

    @staticmethod
    def argmaxmax(my_array, axis=None):
        """
        Compute both argmax (key) and max (value) of a dictionary
        """
        retval_argmax = my_array.argmax(axis=axis)
        if axis is None:
            indexer = unravel_index(retval_argmax, my_array.shape)
        else:
            indexer = [arange(i) for i in my_array.shape]
            indexer[axis] = retval_argmax
        return (retval_argmax, my_array[indexer])

    def _feature_tag(self, sent_efs):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples
        """
        # nomenclature:
        # emission weights: weight for word-given-tag, not taking any
        # context into account
        # transition weights: weight for coming into a state at time `t`
        # state weights: weight for _being_ in a state at time `t`,
        # the sum of emission weights and trellis weights
        L = len(sent_efs)  # len of sentence, in tokens
        if L == 0:
            return []
        # initialize matrix of backpointers
        bckptrs = zeros((L, len(self.idx2tag)), dtype=uint16)
        # special case for first state: no weights from previous states,
        # and no need for transition weights because the "start" features
        # are present in the `w-1` and `w-2` emission features, so
        # state weights are just emission weights
        t = 0
        s_weights = self._e_weights(sent_efs[t])
        if L == 1:
            return [self.idx2tag[s_weights.argmax()]]
        # special case for the second state: we do not need trigram
        # transition weights because the same weights are found in the
        # `w-2` emission feature.
        t += 1
        # add in bigram transition weights and compute max
        (bckptrs[t, ], t_weights) = PPN.argmaxmax(s_weights +
                                                  self.btf_weights, axis=1)
        # combine previous state, transition, and emission weights
        s_weights = t_weights + self._e_weights(sent_efs[t])
        for (t, token_efs) in enumerate(sent_efs[2:], 2):
            # make copy of bigram transition weights matrix
            tf_weights = copy(self.btf_weights)
            # add trigram transition weights to copy
            for (i, prev_idx) in enumerate(bckptrs[t - 1, ]):
                col = tf_weights[:, i]
                trigram_tfs = trigram_tf(self.idx2tag[bckptrs[t - 2, i]],
                                         self.idx2bigram_tfs[prev_idx])
                for (tag, weight) in self.weights[trigram_tfs].iteritems():
                    col[self.tag2idx[tag]] += weight.get(self.time)
            # add in transition weights and compute max
            (bckptrs[t, ], t_weights) = PPN.argmaxmax(s_weights +
                                                      tf_weights, axis=1)
            # combine previous state, transition, and emission weights
            s_weights = t_weights + self._e_weights(token_efs)
        # trace back to get best path
        max_index = s_weights.argmax()
        tags = [self.idx2tag[max_index]]
        while t > 0:
            max_index = bckptrs[t, max_index]
            tags.append(self.idx2tag[max_index])
            t -= 1
        tags.reverse()  # NB: works in place
        return tags

    def tag(self, tokens):
        """
        Tag a single `sentence` list(str)
        """
        return zip(tokens, self._feature_tag(extract_sent_efs(tokens)))

    def batch_tag(self, sentences):
        """
        Tag a list(list(str)) of `sentences`
        """
        for tokens in sentences:
            return zip(tokens, self._feature_tag(extract_sent_efs(tokens)))

    def evaluate(self, sentences):
        """
        Compute tag accuracy of the current model using a held-out list of
        `sentence`s (list of token/tag pairs)
        """
        total = 0
        correct = 0
        for sentence in sentences:
            (tokens, gtags) = zip(*sentence)
            htags = [tag for (_, tag) in self.tag(tokens)]
            for (htag, gtag) in zip(htags, gtags):
                total += 1
                correct += (htag == gtag)
        return correct / total

    def evaluate_sentence(self, sentences):
        """
        Compute _sentence_ accuracy (à la Manning 2011) of the current
        model using a held-out list of `sentence`s (lists of token/tag
        pairs)
        """
        total = 0
        correct = 0
        for sentence in sentences:
            (tokens, gtags) = zip(*sentence)
            htags = [tag for (_, tag) in self.tag(tokens)]
            total += 1
            correct += (htags == gtags)
        return correct / total


if __name__ == '__main__':

    from sys import argv
    from gzip import GzipFile
    from nltk import str2tuple, tuple2str
    from getopt import getopt, GetoptError

    # helpers

    @Listify
    def tag_reader(filename):
        """
        Return a list(list(str, str))) in which the inner list contains
        word/tag tuples representing a single sentence
        """
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wt) for wt in line.strip().split()]

    @Listify
    def untagged_reader(filename):
        """
        Return a list(list(str))) in which the inner list contains a list
        of words (it is assumed there are no tags; if there are tags but
        you wish to ignore them, use `untag`)
        """
        with open(filename, 'r') as source:
            for line in source:
                yield line.strip().split()

    # parse arguments
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

    # check outputs
    if not any((model_sink, untagged_source, test_source)):
        logging.error('No outputs specified.')
        exit(USAGE)

    # run inputs
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

    # run outputs
    if test_source:
        logging.info('Evaluating on data from "{}".'.format(test_source))
        try:
            with open(test_source, 'r') as source:
                print 'Accuracy: {:.04f}'.format(ppn.evaluate(
                                                 tag_reader(test_source)))
        except IOError as err:
            logging.error(err)
            exit(1)
    if untagged_source:
        logging.info('Tagging data from "{}".'.format(untagged_source))
        try:
            for tokens in untagged_reader(untagged_source):
                print ' '.join(tuple2str(wt) for wt in ppn.tag(tokens))
        except IOError as err:
            logging.error(err)
            exit(1)
    if model_sink:
        logging.info('Writing serialized data to "{}".'.format(model_sink))
        try:
            with GzipFile(model_sink, 'w') as sink:
                ppn.dump(sink)
        except IOError as err:
            logging.error(err)
            exit(1)
