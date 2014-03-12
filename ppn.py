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
# timeit tests suggest that for generating a random list of indices of the
# sort used to randomize order of presentation, this is much faster than
# `random.shuffle`; if for some reason you are unable to deploy `numpy`,
# it should not be difficult to modify the code to use `random` instead.
from numpy import arange, array, int16, ones, unravel_index

from decorators import Listify
from lazyweight import LazyWeight
from features import bigram_tf, extract_sent_efs, extract_sent_tfs, \
                                extract_token_tfs, trigram_tf

## defaults and (pseudo)-globals
INF = float('inf')
VERSION_NUMBER = 0.5
TRAINING_ITERATIONS = 10

# set jsonpickle to do it human-readable
jsonpickle.set_encoder_options('simplejson', indent=' ')

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

        -t t           number of training iterations (default: {2})
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
    """
    
    def __repr__(self):
        return '{}(time = {})'.format(self.__class__.__name__, self.time)

    def __init__(self, sentences=None, T=1):
        """
        Initialize the model, if a list(list(str, str)) of `sentences` is
        provided, perform `T` epochs of training
        """
        self.time = 0
        # the outer keys are tags; the outer values are dictionaries with
        # (inner) feature keys and LazyWeight values
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
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
        retval._update_cache()
        return retval

    def dump(self, sink):
        """
        Serialize object (as JSON) and write to `sink`
        """
        print >> sink, jsonpickle.encode(self, keys=True)

    @staticmethod
    @Listify
    def extract_corpus_fs(sentences):
        """
        Extract all features used in training, returning a (tag list,
        token feature list, and tag feature list) tuple for each sentence
        """
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            yield (tags, extract_sent_efs(tokens), extract_sent_tfs(tags))

    def _update_cache(self):
        """
        cache mapping from integer index to tag, and strings representing bigram tag features
        """
        self.index2tag = {i: tag for (i, tag) in enumerate(self.weights)}
        self.btf_strings = [bigram_tf(tag) for tag in self.weights]

    def train(self, sentences, T=1):
        """
        Perform `T` epochs of training using data from `sentences`, a
        list(list(str, str)), as generated by `__main__.tag_reader`
        """
        logging.info('Extracting input features for training.')
        # cache training data features
        corpus_fs = PPN.extract_corpus_fs(sentences)
        # freshen dictionary
        for (tags, _, _) in corpus_fs:
            for tag in tags:
                self.weights[tag]
        # cache tag number and bigram tag feature strings
        self._update_cache()
        # begin training
        logging.info('Beginning {} epochs of training...'.format(T))
        for t in xrange(1, T + 1):
            tic = time()
            epoch_right = epoch_wrong = 0
            for (g_tags, sent_efs, sent_tfs) in permutation(corpus_fs):
                # compare hypothesized tagging to gold standard
                h_tags = self._feature_tag(sent_efs)
                for (h_tag, g_tag, token_efs, token_tfs) in zip(h_tags,
                                              g_tags, sent_efs, sent_tfs):
                    if h_tag == g_tag:
                        epoch_right += 1
                        continue
                    token_fs = token_efs + token_tfs
                    self._update(g_tag, token_fs, +1)  # reward
                    self._update(h_tag, token_fs, -1)  # punish
                    epoch_wrong += 1
                self.time += 1
            # compute accuracy
            acc = epoch_right / (epoch_right + epoch_wrong)
            logging.info('Epoch {:02} acc.: {:.04f}'.format(t, acc) +
                         ' ({}s elapsed).'.format(int(time() - tic)))
        logging.info('Training complete.')

    def _update(self, tag, token_fs, sgn):
        """
        Apply update ("reward" if `sgn` == 1, "punish" if `sgn` == -1) for
        each non-null feature for this `tag`
        """
        tagptr = self.weights[tag]
        for token_f in token_fs:
            tagptr[token_f].update(self.time, sgn)

    def _emission_weights(self, token_efs):
        """
        Use a vector of token features (representing a single state) to
        compute emission weights for this state
        """
        return array([sum(tag_ws[f].get(self.time) for f in token_efs) \
                             for tag_ws in self.weights.itervalues()])

    @staticmethod
    def maxargmax(my_array, axis=None):
        """
        Compute both max and argmax of an numpy array along specified axis
        """
        retval_argmax = my_array.argmax(axis)
        if axis is None: # axis unspecified
            indexer = unravel_index(retval_argmax, my_array.shape)
        else: # axis is specified
            indexer = [arange(i) for i in my_array.shape]
            indexer[axis] = retval_argmax
        return (my_array[indexer], retval_argmax)

    def _feature_tag(self, sent_efs):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples
        """
        # FIXME dispatch these to greedy tagging (for debugging only)
        #return self._feature_tag_greedy(sent_efs)
        # /FIXME
        L = len(sent_efs)  # len of sentence, in tokens
        if L == 0:
            return []
        # initialize matrix of backpointers
        bckptrs = -ones((L, len(self.weights)), dtype=int16)
        ## special case for first state: just emission weights and no
        ## backpointers; there are no weights from previous states, and
        ## there is no need for transition weights because the same
        ## weights are present in the `w-1` and `w-2` emission features
        t = 0
        state_weights = self._emission_weights(sent_efs[t])
        if L == 1:
            return [self.index2tag[state_weights.argmax()]]
        # compute (dense) bigram transition matrix
        btf_matrix = array([[tag_ws[btf_string].get(self.time) for      \
                                    btf_string in self.btf_strings] for \
                                    tag_ws in self.weights.itervalues()])
        ## special case for the second state: we do not need trigram
        ## transition weights because the same weights are found in the
        ## `w-2` emission feature.
        # combine weight from previous state with bigram transition weights
        # and compute the max for each current state, storing backpointers
        t += 1
        (tx_weights, bckptrs[t, ]) = PPN.maxargmax(state_weights +
                                                   btf_matrix, axis=1)
        # combine previous state, transition, and emission weights
        state_weights = tx_weights + self._emission_weights(sent_efs[t])
        ## general case
        for (t, token_efs) in enumerate(sent_efs[2:], 2):
            # combine bigram and trigram transition weights, using a
            # horrendously complex vectorization...explanation follows...
            # Each cell in the `btf_matrix` represents a transition from
            # the previous tag "t - 1" to the current tag "t". For each
            # cell in this matrix, we first determine the most likely
            # "t - 2" (two tags back) given a hypothesized "t - 1" and "t".
            # The index of this tag, `prev_prev_index`, is found in
            # `bckptrs[t - 1,]` and its string can be looked up in
            #  `self.index2tag`. We use this information and the function
            # `trigram_tf` to create the trigram context feature string
            # for each cell in the bigram transition matrix. Then, we
            # look up the weight of that feature (with respect to outcome
            # "t") and add that to the bigram transition cell's weight.
            # The resulting `tf_matrix` contains all the transition
            # weight information we need.
            tf_matrix = btf_matrix + array([[                             \
                        tag_ws[trigram_tf(self.index2tag[prev_prev_index],\
                        btf_string)].get(self.time) for                   \
                        (prev_prev_index, btf_string) in                  \
                        zip(bckptrs[t - 1, ], self.btf_strings)] for      \
                        tag_ws in self.weights.itervalues()])
            # combine weight from previous state with transition weights
            # and compute the max for each current state, storing
            # backpointers
            (tx_weights, bckptrs[t, ]) = PPN.maxargmax(state_weights +
                                                       tf_matrix, axis=1)
            # combine previous state, transition, and emission weights
            state_weights = tx_weights + self._emission_weights(token_efs)
        # trace back to get best path
        max_index = state_weights.argmax()
        tags = [self.index2tag[max_index]]
        while t > 0:
            max_index = bckptrs[t, max_index]
            tags.append(self.index2tag[max_index])
            t -= 1
        tags.reverse()  # NB: works in place
        return tags

    # deprecated: for debugging only
    def _feature_tag_greedy(self, sent_efs):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples
        """
        tags = []
        for token_efs in sent_efs:
            best_tag = None
            best_weight = -INF
            # combine token feature vector with transition features
            token_fs = token_efs + extract_token_tfs(*tags[-2:])
            for (tag, tag_ws) in self.weights.iteritems():
                weight = sum(tag_ws[f].get(self.time) for f in token_fs)
                if weight > best_weight:
                    best_weight = weight
                    best_tag = tag
            tags.append(best_tag)
        return tags

    def tag(self, tokens):
        """
        Tag a single list(str) `sentence`
        """
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
    from getopt import getopt, GetoptError

    # helpers

    def str2tuple(string):
        """
        Convert 'token/tag' string to (token, tag) tuple
        """
        i = string.rfind('/')
        return (string[:i], string[i + 1:])

    def tuple2str(wt_tuple):
        """
        Convert (str, str) representing token/tag pair to string
        """
        return '{}/{}'.format(*wt_tuple)

    def untag(sentence):
        """
        Return a list of tokens extracted from a iterator of (token, tag)
        tuples
        """
        return [w for (w, _) in sentence]

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
        (optlist, args) = getopt(argv[1:], 'i:p:D:E:T:t:chv')
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
        # chunk FIXME
        elif opt == '-c':
            raise NotImplementedError
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
                accuracy = ppn.evaluate(tag_reader(test_source))
                print 'Accuracy: {:.04f}'.format(accuracy)
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
        logging.info('Writing serialized data to "{}".'.format(model_sink))
        try:
            with GzipFile(model_sink, 'w') as sink:
                ppn.dump(sink)
        except IOError as err:
            logging.error(err)
            exit(1)
