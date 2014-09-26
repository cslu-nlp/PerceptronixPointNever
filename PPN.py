#!/usr/bin/env python
#
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
# ppn.py: Perceptronix Point Never, a perceptron-backed POS tagger


import logging

from time import time
from functools import lru_cache
from nltk import str2tuple, tuple2str

from jsonable import JSONable
from confusion import Accuracy
from decorators import IO, listify
from perceptron import SequenceAveragedPerceptron as SequenceClassifier


EPOCHS = 10
ORDER = 2

LPAD = ["<S1>", "<S0>"]
RPAD = ["</S0>", "</S1>"]

LOGGING_FMT = "%(message)s"


# helpers


@IO
@listify
def tagged_corpus(filename):
    """
    Read tagged corpus into memory
    """
    with open(filename, "r") as source:
        for line in source:
            yield [str2tuple(wt) for wt in line.split()]


@IO
@listify
def untagged_corpus(filename):
    """
    Read tokenized, but untagged, corpus into memory
    """
    with open(filename, "r") as source:
        for line in source:
            yield line.split()

# feature extractors


@lru_cache(128)
def fstring(key, value):
    return "{}='{}'".format(key, value)


@listify
def efeats(tokens, order=ORDER):
    """
    Compute list of lists of emission features for each token in 
    the `tokens` iterator
    """
    padded_tokens = LPAD + [t.upper() for t in tokens] + RPAD
    for (i, token) in enumerate(padded_tokens[2:-2], 2):
        feats = ["(bias)"]
        # adjacent tokens
        for j in range(-order, order + 1):
            feats.append(fstring("w_i{:+d}".format(j),
                                 padded_tokens[i + j]))
        # orthographic matters
        if token.isdigit():
            feats.append("(digits)")
        if token.istitle():
            feats.append("(titlecase)")
            if token.isupper():
                feats.append("(uppercase)")
        if "-" in token:
            feats.append("(hyphen)")
        if "'" in token:
            feats.append("(apostrophe)")
        for i in range(1, 1 + min(len(token) - 1, 4)):
            feats.append(fstring("prefix({})".format(i), token[:+i]))
            feats.append(fstring("suffix({})".format(i), token[-i:]))
        yield feats


def tfeats(tags):
    """
    Compute a list of features for a single token using an iterator of
    tags; this also partially determines the Markov order. An example:

    >>> d = 3
    >>> tags = "RB DT JJ NN".split()
    >>> sorted(tfeats(tags[:1]))
    ["t_i-1='RB'"]
    >>> sorted(tfeats(tags[:3]))
    ["t_i-1='JJ'", "t_i-2,t_i-1='DT','JJ'", "t_i-3,t_i-2,t_i-1='RB','DT','JJ'"]
    """
    feats = []
    if not tags:
        return feats
    i = 1
    tfeat_key = "t_i-{}".format(i)
    feats.append(fstring(tfeat_key, tags[-i]))
    for i in range(2, 1 + len(tags)):
        tfeat_key = "t_i-{},{}".format(i, tfeat_key)
        vstring = ",".join("'{}'".format(tag) for tag in tags[-i:])
        feats.append("{}={}".format(tfeat_key, vstring))
    return feats


class Tagger(JSONable):

    """
    Part-of-speech tagger, backed by a classifier
    """

    def __init__(self, *, tfeats_fnc=tfeats, order=ORDER, epochs=EPOCHS,
                 sentences):
        self.classifier = SequenceClassifier(tfeats_fnc=tfeats_fnc,
                                             order=order)
        if sentences:
            self.fit(sentences, epochs=epochs)

    def fit(self, sentences, epochs=EPOCHS):
        XX = []
        YY = []
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            XX.append(efeats(tokens))
            YY.append(list(tags))
        self.classifier.fit(XX, YY, epochs)

    @listify
    def tag(self, tokens):
        xx = efeats(tokens)
        return zip(tokens, self.classifier.predict(xx))

    @listify
    def batch_tag(self, tokens_list):
        for tokens in tokens_list:
            yield self.tag(tokens)


if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser(
        description="Perceptronix Point Never, by Kyle Gorman")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           help="enable verbose output")
    argparser.add_argument("-V", "--really-verbose", action="store_true",
                           help="even more verbose output")
    input_group = argparser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--train", help="training data")
    input_group.add_argument("-r", "--read",
                             help="read in serialized model")
    output_group = argparser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("-u", "--tag", help="tag unlabeled data")
    output_group.add_argument("-w", "--write",
                              help="write out serialized model")
    output_group.add_argument("-e", "--evaluate",
                              help="evaluate on labeled data")
    argparser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
                           help="# of epochs (default: {})".format(EPOCHS))
    argparser.add_argument("-O", "--order", type=int, default=ORDER,
                           help="Markov order (default: {})".format(ORDER))
    args = argparser.parse_args()
    # verbosity block
    if args.verbose:
        logging.basicConfig(format=LOGGING_FMT, level=logging.INFO)
    elif args.really_verbose:
        logging.basicConfig(format=LOGGING_FMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOGGING_FMT)
    # check for valid output
    if not (args.tag or args.write or args.evaluate):
        argparser.error("No outputs specified.")
    # input
    tagger = None
    if args.train:
        logging.info("Training on labeled data '{}'.".format(args.train))
        sentences = tagged_corpus(args.train)
        tagger = Tagger(epochs=args.epochs, order=args.order,
                        sentences=sentences)
    elif args.read:
        logging.info("Reading pretrained tagger '{}'.".format(args.read))
        tagger = IO(Tagger.load)(args.read)
    # else unreachable
    # output
    if args.tag:
        logging.info("Tagging untagged data '{}'.".format(args.tag))
        for tokens in untagged_corpus(args.tag):
            print(" ".join(tuple2str(wt) for wt in tagger.tag(tokens)))
    elif args.write:
        logging.info("Writing trained tagger to '{}'.".format(args.write))
        IO(tagger.dump)(args.write)
    elif args.evaluate:
        logging.info("Evaluating tagged data '{}'.".format(args.evaluate))
        accuracy = Accuracy()
        for sentence in tagged_corpus(args.evaluate):
            (tokens, tags) = zip(*sentence)
            tags_guessed = (tag for (token, tag) in tagger.tag(tokens))
            accuracy.batch_update(tags, tags_guessed)
        print("Accuracy: {:.04f}".format(accuracy.accuracy))
    # else unreachable
