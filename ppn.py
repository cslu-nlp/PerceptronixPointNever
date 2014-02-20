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
#
# TODO:
#   * optimize out tag feature computation
#   * only keep two rows of the trellis
#   * add IOB chunking support
#   * port to Python 3 (once NLTK support is available?)

from __future__ import division

import logging
import jsonpickle

from time import time
from numpy import ones, zeros
from numpy.random import permutation
from collections import deque, defaultdict
# timeit tests suggest that for generating a random list of indices of the
# sort used to randomize order of presentation, this is much faster than
# `random.shuffle`; if for some reason you are unable to deploy `numpy`,
# it should not be difficult to modify the code to use `random` instead.

from features import *
from lazyweight import LazyWeight
from decorators import Listify, Setify

## defaults and (pseudo)-globals
VERSION_NUMBER = 0.3
TRAINING_ITERATIONS = 10
INF = float('inf')

# set jsonpickle to do it human-readable
jsonpickle.set_encoder_options('simplejson', indent=' ')

# usage string
USAGE = """Perceptronix Point Never {}, by Kyle Gorman and Steven Bedrick

    python ppn.py [-i tag] [-i|-p input] [-D|-E|-T output] [-t t] [-h] [-v]

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
        return jsonpickle.decode(source.read(), keys=True)

    def dump(self, sink):
        """
        Serialize object (as JSON) and write to `sink`
        """
        print >> sink, jsonpickle.encode(self, keys=True)

    @staticmethod
    @Listify
    def _get_features(sentences):
        """
        Extract all features used in training, returning a (tag list,
        token feature list, and tag feature list) tuple for each sentence
        """
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            yield tags, POS_token_features(tokens), POS_tag_features(tags)

    @staticmethod
    def _get_index2tag(sentence_features):
        """
        Construct dictionary mapping from a counting number index to the
        associated tag
        """
        tagset = set()
        for (tags, _, _) in sentence_features:
            tagset.update(tags)
        return {i: tag for (i, tag) in enumerate(tagset)}

    def train(self, sentences, T=1):
        logging.info('Extracting input features for training.')
        # each element of the `sentfeats` represents a sentence:
        #   each sentence is a triple of tags, token feature lists,
        #   and tag feature lists
        sentfeats = PPN._get_features(sentences)
        # construct dictionary mapping from tag to index in trellis
        self.index2tag = PPN._get_index2tag(sentfeats)
        # begin training
        logging.info('Beginning {} epochs of training...'.format(T))
        for t in xrange(1, T + 1):
            tic = time()
            epoch_right = epoch_wrong = 0
            for (gtags, tokfeats, tagfeats) in permutation(sentfeats):
                # compare hypothesized tagging to gold standard
                htags = self._feature_tag(tokfeats)
                for (htag, gtag, tokf, tagf) in zip(htags, gtags, tokfeats,                                                                 tagfeats):
                    if htag == gtag:
                        epoch_right += 1
                        continue
                    feats = tokf + tagf
                    self._update(gtag, feats, +1)
                    self._update(htag, feats, -1)
                    epoch_wrong += 1
                self.time += 1
            # compute accuracy
            acc = epoch_right / (epoch_right + epoch_wrong)
            logging.info('Epoch {:02} acc.: {:.04f}'.format(t, acc) +
                         ' ({}s elapsed).'.format(int(time() - tic)))
        logging.info('Training complete.')

    def _update(self, tag, featset, sgn):
        """
        Apply update ("reward" if `sgn` == 1, "punish" if `sgn` == -1) for
        each feature in `features` for this `tag`
        """
        tagptr = self.weights[tag]
        for feat in featset:
            tagptr[feat].update(self.time, sgn)

    def tag(self, tokens):
        """
        Tag a single `sentence` (list of tokens)
        """
        return zip(tokens, self._feature_tag(POS_token_features(tokens)))

    def _feature_tag_greedy(self, tokfeats):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples

        This is deprecated; use Viterbi (_feature_tag) instead. It's not
        called by any code, but is nice to keep around for debugging.
        """
        tags = []
        for i, tokfeatset in enumerate(tokfeats):
            featset = tokfeatset + tag_featset(*tags[-2:])
            best_tag = None
            best_weight = -INF
            for tag in self.index2tag.itervalues():
                tagptr = self.weights[tag]
                weight = sum(tagptr[feat].get(self.time) for
                             feat in featset)
                if weight > best_weight:
                    best_weight = weight
                    best_tag = tag
            tags.append(best_tag)
        return tags

    def _feature_tag(self, tokfeats):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples
        """
        L = len(tokfeats)  # len of sentence, in tokens
        if L == 0:
            return []
        Lt = len(self.index2tag)
        trellis = zeros((L, Lt), dtype=int)
        bckptrs = -ones((L, Lt), dtype=int)
        # special case for first tag position
        featset = tokfeats[0] + tag_featset(*LEFT_PAD)
        for (i, tag) in self.index2tag.iteritems():
            tagptr = self.weights[tag]
            trellis[0, i] = sum(tagptr[feat].get(self.time) for
                                feat in featset)
        # if there's only one word, we need to halt now
        if L == 1:
            return [self.index2tag[trellis[0, ].argmax()]]
        # special case for second tag position
        for (i, tag) in self.index2tag.iteritems():
            tagptr = self.weights[tag]
            best_transition = -1
            best_weight = -INF
            for (j, prev_tag) in self.index2tag.iteritems():
                tagfeatset = tag_featset(LEFT_PAD[1], prev_tag)
                transition_weight = sum(tagptr[feat].get(self.time) for
                                        feat in tagfeatset)
                state_weight = trellis[0, j] + transition_weight
                if state_weight > best_weight:
                    best_weight = state_weight
                    best_transition = j
            emission_weight = sum(tagptr[feat].get(self.time) for
                                  feat in tokfeats[1])
            trellis[1, i] = emission_weight + best_weight
            bckptrs[1, i] = best_transition
        # all other tag positions
        for t in xrange(2, L):
            for (i, tag) in self.index2tag.iteritems():  # for each tag
                tagptr = self.weights[tag]
                best_transition = -1
                best_weight = -INF
                for (j, prev_tag) in self.index2tag.iteritems():
                    prev_prev_tag = self.index2tag[bckptrs[t - 1, j]]
                    tagfeatset = tag_featset(prev_prev_tag, prev_tag)
                    transition_weight = sum(tagptr[feat].get(self.time) for
                                            feat in tagfeatset)
                    state_weight = trellis[t - 1, j] + transition_weight
                    if state_weight > best_weight:
                        best_weight = state_weight
                        best_transition = j
                emission_weight = sum(tagptr[feat].get(self.time) for
                                      feat in tokfeats[t])
                trellis[t, i] = emission_weight + best_weight
                bckptrs[t, i] = best_transition
        t = L - 1
        # get index of best final state/tag
        index = trellis[t, ].argmax()
        # store it in a LIFO queue
        tags = deque([self.index2tag[index]])
        # follow backptrs from there
        while t > 0:
            index = bckptrs[t, index]
            tags.appendleft(self.index2tag[index])
            t -= 1
        return list(tags)

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
    from getopt import getopt, GetoptError

    from decorators import Listify

    # helpers

    def str2tuple(s):
        i = s.rfind('/')
        return (s[:i], s[i + 1:])

    def untag(sentence):
        return [w for (w, t) in sentence]

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
