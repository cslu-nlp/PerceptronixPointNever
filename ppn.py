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
# Perceptronix Point Never: an HMM sequence tagger trained using the
# averaged perceptron algorithm

from __future__ import division

import logging
import jsonpickle

from time import time
from string import digits
from gzip import GzipFile
from collections import defaultdict

from numpy.random import permutation  # faster than `random.shuffle`
from numpy import arange, uint16, unravel_index, zeros

from nltk import str2tuple, tuple2str, TaggerI, ChunkParserI, ChunkScore, \
                                                                    Tree

## defaults and globals
VERSION_NUMBER = 0.8
TRAINING_ITERATIONS = 10

USAGE = """Perceptronix Point Never {0}, by Kyle Gorman and Steven Bedrick

    USAGE: {1} [-i|-p] file [-I|-P|-E] file [-T {2}] [-c] [-h] [-v]

    Input arguments (at least one required):

        -i file        train on labeled data
        -p file        read in serialized model

    Output arguments (at least one required):

        -I file        tag or chunk unlabeled data `unlabeled
        -E file        evaluate on labeled data
        -P file        write out serialized model
    
    Optional arguments:

        -T iters       number of training iterations       [default: {2}]
        -c             run as chunker, not as a tagger     [default: {3}]

        -h             print this message and quit
        -v             increase verbosity

    All inputs should be whitespace-delimited with one sentence per line.

    Tagger training/evaluation: "token/POS-tag"
    Tagging: bare tokens (no POS tags)

    Chunker training/evaluation: "token/POS-tag/chunk-tag"
    Chunking: "token/POS-tag" tokens (no chunk tags)
""".format(VERSION_NUMBER, __file__, TRAINING_ITERATIONS, 'no')

# lazy weight class


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
    value of the summed weight (for inference), we "freshen" the summed
    weight by adding to it the product of the real weight and the time
    elapsed.

    While passing around the "timer" of the outer class is suboptimal, one
    advantage of this format is that we can store weights and their times
    in the same place, reducing the number of redundant hashtable lookups
    required.

    # initialize
    >>> t = 0
    >>> lw = LazyWeight(1, t)
    >>> lw.get(t)
    1

    # some time passes...
    >>> t += 1
    >>> lw.get(t)
    2

    # weight is now changed
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.update(-1, t)
    >>> t += 3
    >>> lw.get(t)
    -1
    """

    def __init__(self, weight=0, time=0):
        self.weight = self.summed_weight = weight
        self.timestamp = time

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def _freshen(self, time):
        """
        Apply queued updates and freshen the timestamp
        """
        if time == self.timestamp:
            return
        self.summed_weight += (time - self.timestamp) * self.weight
        self.timestamp = time

    def get(self, time):
        """
        Return an up-to-date sum of weights
        """
        self._freshen(time)
        return self.summed_weight

    def update(self, value, time):
        """
        Bring sum of weights up to date, then add `value` to the weight
        """
        self._freshen(time)
        self.weight += value

# averaged perceptron sequence tagger class


class PPN(object):

    """
    Perceptronix Point Never: an HMM sequence tagger with discriminative
    training via the averaged perceptron algorithm

    This lacks emission feature extractors. By mixing them in, and adding
    evaluation methods, you can create a POS tagger or a phrase chunker.
    """

    def __repr__(self):
        return '{}(time = {})'.format(self.__class__.__name__, self.time)

    def __init__(self, sentences=None, T=1):
        """
        Initialize the model; if a list(list(str, str)) of `sentences` is
        provided, perform `T` epochs of training
        """
        self.time = 0
        self.tagset = set()
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
        logging.info('Constructed new PPN instance.')
        if sentences:
            self.train(sentences, T)

    # alternative constructor using serialized JSON

    @classmethod
    def load(cls, filename):
        """
        Deserialize from compressed JSON in `source`, and update all caches
        """
        with GzipFile(filename, 'r') as source:
            retval = jsonpickle.decode(source.read(), keys=True)
        # for some reason self.weights.default_factory does not persist
        retval.weights.default_factory = lambda: defaultdict(LazyWeight)
        # update caches
        retval._update_tagset_cache()
        retval._update_transition_cache()
        return retval

    def dump(self, filename):
        """
        Serialize to compressed JSON in `sink`
        """
        with GzipFile(filename, 'w') as sink:
            print >> sink, jsonpickle.encode(self, keys=True)

    # transition feature extractors

    def _extract_sent_tfs(self, tags):
        """
        Extract bigram and trigram transition features for a list of tags
        representing a single sentence. There is one list of transition
        feature strings per tag.

        :type tags: list(str)
        :param tags: List of part-of-speech tags.
        :rtype: list(list(str))
        """
        # for the first two tokens, there are no tag features; these would
        # recapitulate information in the word features (e.g., w-1) anyways
        for i in xrange(2):
            yield []
        # general case
        for i in xrange(len(tags) - 2):
            yield self._extract_token_tfs(tags[i], tags[i + 1])

    def _extract_token_tfs(self, prev_prev_tag, prev_tag):
        """
        Extract bigram and trigram tranisition features for a single tag.
        Both arguments are strings.

        :type prev_prev_tag: str
        :param prev_prev_tag: the tag two tags back
        :type prev_tag: str
        :param prev_tag: the previous tag
        :rtype: list(str)
        """
        bigram_tf_string = self._bigram_tf(prev_tag)
        trigram_tf_string = self._trigram_tf(prev_prev_tag,
                                             bigram_tf_string)
        return [bigram_tf_string, trigram_tf_string]

    def _bigram_tf(self, prev_tag):
        """
        Create bigram transition feature string

        :type prev_tag: str
        :param prev_tag: the previous tag
        :rtype: str
        """
        return "t-1='{}'".format(prev_tag)

    def _trigram_tf(self, prev_prev_tag, bigram_feature_string):
        """
        Create trigram transition feature string

        :type prev_prev_tag: str
        :param prev_prev_tag: the tag two tags back
        :type bigram_feature_string: str
        :param bigram_feature_string: a bigram feature string for the
                                      previous tag, created with bigram_tf
        """
        return "t-2='{}',{}".format(prev_prev_tag, bigram_feature_string)

    # emission feature class constants

    LPAD = ["<S1>", "<S0>"]
    RPAD = ["</S0>", "</S1>"]

    # ordinary instance methods (and helpers)

    def tic(self):
        """
        Increment the (notional) clock, and update transition cache.
        """
        self.time += 1
        self._update_transition_cache()

    def _update_tagset_cache(self, corpus_tags=None):
        """
        Update the cache of tagset information (and also the transition
        cache, which uses the relevant indices).
        """
        # map between tag, matrix index, and bigram feature string
        self.idx2tag = {i: tag for (i, tag) in enumerate(self.tagset)}
        self.tag2idx = {tag: i for (i, tag) in self.idx2tag.iteritems()}
        self.idx2bigram_tfs = {i: self._bigram_tf(prev_tag) for
                               (i, prev_tag) in self.idx2tag.iteritems()}

    def _update_transition_cache(self):
        """
        Update the cache of bigram transition weights.
        """
        self.Lt = len(self.tagset)
        self.btf_weights = zeros((self.Lt, self.Lt), dtype=int)
        # tags at t are the first axis, tags at t - 1 are the second
        for (i, bigram_tfs) in self.idx2bigram_tfs.iteritems():
            col = self.btf_weights[:, i]
            for (tag, weight) in self.weights[bigram_tfs].iteritems():
                col[self.tag2idx[tag]] += weight.get(self.time)

    def _extract_corpus_features(self, sentences):
        for sentence in sentences:
            (emissions, hidden_states) = zip(*sentence)
            yield (hidden_states,
                   list(self._extract_sent_efs(emissions)),
                   list(self._extract_sent_tfs(hidden_states)))
            self.tagset.update(hidden_states)

    def _update(self, token_fs, g_tag, h_tag):
        """
        Update weights, rewarding the correct tag `g_tag` and penalizing
        the incorrect tag `h_tag`.
        """
        for feat in token_fs:
            featptr = self.weights[feat]
            featptr[g_tag].update(+1, self.time)
            featptr[h_tag].update(-1, self.time)

    def train(self, sentences, T=1):
        """
        Perform `T` epochs of training using data from `sentences`, a
        list(list(str, str)), as generated by `__main__.tag_reader`.
        """
        logging.info('Extracting input features for training.')
        (corpus_tags, corpus_efs, corpus_tfs) = zip(
            *self._extract_corpus_features(sentences))
        epoch_size = sum(len(sent_tags) for sent_tags in corpus_tags)
        self._update_tagset_cache()
        self._update_transition_cache()
        # begin training
        logging.info('Beginning {} epochs of training...'.format(T))
        for n in xrange(1, T + 1):
            toc = time()
            epoch_wrong = 0
            for (g_tags, sent_efs, sent_tfs) in permutation(zip(
                                   corpus_tags, corpus_efs, corpus_tfs)):
                # generate hypothesized tagging and compare to gold tagging
                h_tags = self._feature_tag(sent_efs)
                for (h_tag, g_tag, token_efs, token_tfs) in zip(h_tags,
                                              g_tags, sent_efs, sent_tfs):
                    if h_tag == g_tag:
                        continue
                    self._update(token_efs + token_tfs, g_tag, h_tag)
                    epoch_wrong += 1
                self.tic()
            # compute accuracy
            accuracy = 1. - (epoch_wrong / epoch_size)
            logging.info('Epoch {:02} '.format(n) +
                         'accuracy: {:.04f} '.format(accuracy) +
                         '({}s elapsed).'.format(int(time() - toc)))
        logging.info('Training complete.')

    def _e_weights(self, token_efs):
        """
        Use a vector of token features (representing a single state) to
        compute emission weights for a state with `token_efs` features.
        """
        e_weights = zeros(self.Lt, dtype=int)
        for feat in token_efs:
            for (tag, weight) in self.weights[feat].iteritems():
                e_weights[self.tag2idx[tag]] += weight.get(self.time)
        return e_weights

    @staticmethod
    def argmaxmax(my_array, axis=None):
        """
        Compute both argmax (key) and max (value) over some axis.
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
        # emission weights: weight for word-given-tag, not taking any
        # context into account
        # transition weights: weight for coming into a state at time `t`
        # state weights: weight for _being_ in a state at time `t`,
        # the sum of emission weights and trellis weights
        L = len(sent_efs)  # sentence length, in tokens
        if L == 0:
            return []
        # initialize matrix of backpointers
        bckptrs = zeros((L, self.Lt), dtype=uint16)
        # special case for first state: no weights from previous states,
        # and no need for transition weights because the "start" features
        # are present in the `w-1` and `w-2` emission features, so
        # state weights are just emission weights
        t = 0
        s_weights = self._e_weights(sent_efs[t])
        if L == 1:
            return [self.idx2tag[s_weights.argmax()]]
        # special case for the second state: we do not need trigram
        # transition weights because the same weights are present in the
        # `w-2` emission feature.
        t += 1
        # add in bigram transition weights and compute max
        (bckptrs[t, ], t_weights) = PPN.argmaxmax(s_weights +
                                                  self.btf_weights, axis=1)
        # combine previous state, transition, and emission weights
        s_weights = t_weights + self._e_weights(sent_efs[t])
        for (t, token_efs) in enumerate(sent_efs[2:], 2):
            # make copy of bigram transition weights matrix
            tf_weights = self.btf_weights.copy()
            # add trigram transition weights to copy
            for (i, prev_idx) in enumerate(bckptrs[t - 1, ]):
                # get view/pointer to present row in the transition matrix
                row = tf_weights[i, :]
                trigram_tfs = self._trigram_tf(
                                   self.idx2tag[bckptrs[t - 2, i]],
                                   self.idx2bigram_tfs[prev_idx])
                for (tag, weight) in self.weights[trigram_tfs].iteritems():
                    row[self.tag2idx[tag]] += weight.get(self.time)
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

    def tag(self, sentence):
        return zip(sentence, self._feature_tag(
            list(self._extract_sent_efs(sentence))))


class PPNTagger(PPN, TaggerI):

    def print_evaluation(self, filename):
        gold = list(self.labeled_reader(filename))
        print 'Accuracy: {:.04}'.format(self.evaluate(gold))

    # corpus readers

    @staticmethod
    def unlabeled_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield line.strip().split()

    @staticmethod
    def labeled_reader(filename, sep='/'):
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wt, sep) for wt in line.strip().split()]

    # labeler

    def label(self, filename, sep='/'):
        for tokens in self.unlabeled_reader(filename):
            print ' '.join(tuple2str(wt, sep) for wt in self.tag(tokens))

    # feature extraction

    PRE_SUF_MAX = 4

    def _extract_sent_efs(self, tokens):
        """
        Extract the Ratnaparkhi/Collins POS tagging emission features for
        a list of tokens representing a single sentence. There is one list
        of emission feature strings per token.

        :type tokens: list(str)
        :param tokens: List of part-of-speech tokens.
        :rtype: list(list(str))
        """
        padded_tokens = self.LPAD + [t.lower() for t in tokens] + self.RPAD
        for (i, ftoken) in enumerate(padded_tokens[2:-2]):
            # NB: the "target" is i + 2
            featset = ['b']  # initialize with bias term
            # tokens nearby
            featset.append("w-2='{}'".format(padded_tokens[i]))
            featset.append("w-1='{}'".format(padded_tokens[i + 1]))
            featset.append("w='{}'".format(ftoken))
            featset.append("w+1='{}'".format(padded_tokens[i + 3]))
            featset.append("w+2='{}'".format(padded_tokens[i + 4]))
            for j in xrange(1, 1 + min(len(ftoken), self.PRE_SUF_MAX)):
                featset.append("pre({})='{}'".format(j, ftoken[:+j]))
                featset.append("suf({})='{}'".format(j, ftoken[-j:]))
            # contains a hyphen?
            if any(c == '-' for c in ftoken):
                featset.append('h')
            # contains a number?
            if any(c in digits for c in ftoken):
                featset.append('n')
            # contains an uppercase character?
            if ftoken != tokens[i]:  # which has no case folding
                featset.append('u')
            yield featset


class PPNChunker(PPN, ChunkParserI):

    @staticmethod
    def to_tree(sentence, chunk_types={'NP'}, root_label='ROOT'):
        """
        Convert tagged list into shallow parse tree; supposedly this is
        what `nltk.chunk.util.conlltags2tree` does, but it appears to be 
        broken (it ignores `chunk_types`)
        """
        tree = Tree(root_label, [])
        for ((token, pos_tag), chunk_tag) in sentence:
            if chunk_tag.startswith('B-'):
                if chunk_tag[2:] in chunk_types:
                    tree.append(Tree(chunk_tag[2:], [(token, pos_tag)]))
                else:
                    tree.append((token, pos_tag))
            elif chunk_tag.startswith('I-'):
                if chunk_tag[2:] in chunk_types:
                    tree[-1].append((token, pos_tag))
                else:
                    tree.append((token, pos_tag))
            else: # chunk_tag == 'O' or not in chunk_types
                tree.append((token, pos_tag))
        return tree

    def parse(self, sentence):
        """
        The sentence is tagged, flattened, then converted to shallow parse
        """
        return PPNChunker.to_tree(self.tag(sentence))

    def print_evaluation(self, filename):
        cnk_score = ChunkScore()
        for sentence in self.labeled_reader(filename):
            gold = PPNChunker.to_tree(sentence)
            hypothesis = self.parse(gold.leaves())
            cnk_score.score(gold, hypothesis)
        print 'Accuracy:  {:.04}'.format(cnk_score.accuracy())
        print 'Precision: {:.04}'.format(cnk_score.precision())
        print 'Recall:    {:.04}'.format(cnk_score.recall())
        print 'F-score:   {:.04}'.format(cnk_score.f_measure())

    # corpus readers

    @staticmethod
    def unlabeled_reader(filename, sep='/'):
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wp, sep) for wp in line.strip().split()]

    @staticmethod
    def labeled_reader(filename, sep='/'):
        with open(filename, 'r') as source:
            for line in source:
                hlf = [str2tuple(wpc, sep) for wpc in line.strip().split()]
                yield [(str2tuple(wp, sep), c) for (wp, c) in hlf]

    # labeler

    def label(self, filename, sep='/'):
        for sentence in self.unlabeled_reader(filename):
            print ' '.join('{0}{3}{1}{3}{2}'.format(w, p, c, sep) for
                           ((w, p), c) in self.tag(sentence))

    # feature extraction

    def _extract_sent_efs(self, tokens_pos_tags):
        """
        Extract the Ratnaparkhi/Collins chunking emission features for
        a list of tokens/tags representing a single sentence. There is
        one list of emission feature strings per token/tag.

        :type tokens: ?.
        :param tokens: ?.
        :rtype: list(list(str))
        """
        (tokens, pos_tags) = zip(*tokens_pos_tags)
        padded_tokens = self.LPAD + [t.lower() for t in tokens] + self.RPAD
        padded_tags = self.LPAD + list(pos_tags) + self.RPAD
        for i in xrange(len(tokens)):
            # NB: the "target" is i + 2
            featset = ['b']
            # token unigrams
            featset.append("w-2='{}'".format(padded_tokens[i]))
            featset.append("w-1='{}'".format(padded_tokens[i + 1]))
            featset.append("w='{}'".format(padded_tokens[i + 2]))
            featset.append("w+1='{}'".format(padded_tokens[i + 3]))
            featset.append("w+2='{}'".format(padded_tokens[i + 4]))
            # token bigrams
            featset.append("w-2='{}',w-1='{}'".format(
                           *padded_tokens[i:i + 2]))
            featset.append("w-1='{}',w='{}'".format(
                           *padded_tokens[i + 1:i + 3]))
            featset.append("w='{}',w+1='{}'".format(
                           *padded_tokens[i + 2:i + 4]))
            featset.append("w+1='{}',w+2='{}'".format(
                           *padded_tokens[i + 3:i + 5]))
            # POS tag unigrams
            featset.append("t-2='{}'".format(padded_tags[i]))
            featset.append("t-2='{}'".format(padded_tags[i + 1]))
            featset.append("t='{}'".format(padded_tags[i + 2]))
            featset.append("t+1='{}'".format(padded_tags[i + 3]))
            featset.append("t+2='{}'".format(padded_tags[i + 4]))
            # POS tag bigrams
            featset.append("t-2='{}',t-1='{}'".format(
                           *padded_tags[i:i + 2]))
            featset.append("t-1='{}',t='{}'".format(
                           *padded_tags[i + 1:i + 3]))
            featset.append("t='{}',t+1='{}'".format(
                           *padded_tags[i + 2:i + 4]))
            featset.append("t+1='{}',t+2='{}'".format(
                           *padded_tags[i + 3:i + 5]))
            # POS tag trigrams
            featset.append("t-2='{}',t-1='{}',t='{}'".format(
                           *padded_tags[i:i + 3]))
            featset.append("t-1='{}',t='{}',t+1='{}'".format(
                           *padded_tags[i + 1:i + 4]))
            featset.append("t='{}',t+1='{}',t+2='{}'".format(
                           *padded_tags[i + 2:i + 5]))
            yield featset


if __name__ == '__main__':

    from sys import argv, stderr
    from getopt import getopt, GetoptError

    def print_usage():
        print >> stderr, USAGE

    # parse arguments
    try:
        (optlist, args) = getopt(argv[1:], 'i:p:I:P:E:T:chv')
    except GetoptError as err:
        print_usage()
        logging.error(err)
        exit(1)
    # warn users about arguments without flags (as this is unsupported)
    for arg in args:
        logging.warning('Ignoring command-line argument "{}"'.format(arg))
    # set defaults
    tagger_mode = True
    train = deserialize = None              # input options
    label = serialize = evaluate = None     # output options
    training_iterations = TRAINING_ITERATIONS
    # read in optlist
    for (opt, arg) in optlist:
        if opt == '-i':
            train = arg
        elif opt == '-p':
            deserialize = arg
        elif opt == '-I':
            label = arg
        elif opt == '-P':
            serialize = arg
        elif opt == '-E':
            evaluate = arg
        elif opt == '-c':
            tagger_mode = False
        elif opt == '-h':
            print_usage()
            exit(1)
        elif opt == '-v':
            logging.basicConfig(level=logging.INFO)
        elif opt == '-T':
            try:
                training_iterations = int(arg)
                if training_iterations < 1:
                    raise ValueError('{} arg must be > 0'.format(opt))
            except ValueError as err:
                logging.error(err)
                exit(1)
        else:
            print_usage()
            logging.error('Option {} not found.'.format(opt, arg))
            exit(1)

    # inputs
    labeler = None
    if deserialize:
        try:
            logging.info('Deserializing "{}".'.format(deserialize))
            labeler = PPN.load(deserialize)
        except IOError as err:
            logging.error(err)
            exit(1)
    if train:
        try:
            if tagger_mode:
                logging.info('Training tagger with "{}".'.format(train))
                labeler_type = PPNTagger
            else:
                logging.info('Training chunker with "{}".'.format(train))
                labeler_type = PPNChunker
            # actually train
            labeler = labeler_type()
            labeler.train(labeler_type.labeled_reader(train),
                          training_iterations)
        except IOError as err:
            logging.error(err)
            exit(1)
    if not labeler:
        print_usage()
        logging.error('No inputs specified.')
        exit(1)

    # outputs
    if label:
        logging.info('Labeling "{}".'.format(label))
        try:
            labeler.label(label)
        except IOError as err:
            logging.error(err)
            exit(1)
    if evaluate:
        try:
            labeler.print_evaluation(evaluate)
        except IOError as err:
            logging.error(err)
            exit(1)
    if serialize:
        try:
            labeler.dump(serialize)
        except IOError as err:
            logging.error(err)
            exit(1)
