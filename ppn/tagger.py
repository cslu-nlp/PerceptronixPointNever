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


import logging

from time import time
from random import Random
from string import punctuation
from functools import lru_cache

from nltk import str2tuple

from nlup import Accuracy, JSONable, IO, listify, \
                 SequenceAveragedPerceptron as SequenceClassifier


EPOCHS = 10
ORDER = 2

LPAD = ["<S1>", "<S0>"]
RPAD = ["</S0>", "</S1>"]

PUNCTUATION = frozenset(punctuation)
NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen
nineteen
twenty thirty fourty fifty sixty seventy eighty ninety 
hundred thousand million billion trillion
""".upper().split())



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


def isnumberlike(token):
    # remove ',' and '.'
    token = token.replace(".", "").replace(",", "")
    # generic digit
    if token.isdigit():
        return True
    # fraction
    if token.count("/") == 1:
        (numerator, denominator) = token.split("/", 1)
        if numerator.isdigit() and denominator.isdigit():
            return True
    # number words
    if token in NUMBER_WORDS:
        return True
    return False


@listify
def efeats(tokens, order=ORDER):
    """
    Compute list of lists of emission features for each token in 
    the `tokens` iterator
    """
    padded_tokens = LPAD + [t.upper() for t in tokens] + RPAD
    for (i, token) in enumerate(padded_tokens[2:-2], 2):
        feats = ["*bias*"]
        # adjacent tokens
        for j in range(-order, order + 1):
            feats.append(fstring("w_i{:+d}".format(j),
                                 padded_tokens[i + j]))
        # orthographic matters
        if token.isdigit():
            feats.append("*digits*")
        else:
            if token.islower():
                feats.append("*lowercase*")
            elif token.isupper():
                feats.append("*uppercase*")
            elif token.istitle():
                feats.append("*titlecase*")
        if isnumberlike(token):
            feats.append("*numberlike*")
        if all(char in PUNCTUATION for char in token):
            feats.append("*punctuation*")
        if "-" in token:
            feats.append("*hyphen*")
        #if "'" in token:
        #    feats.append("(apostrophe)")
        # prefixes and suffixes
        feats.append(fstring("pre1(w_i+0)", token[:1]))
        feats.append(fstring("suf3(w_i-1)", padded_tokens[i - 1][-3:]))
        feats.append(fstring("suf3(w_i+0)", token[-3:]))
        feats.append(fstring("suf3(w_i+1)", padded_tokens[i + 1][-3:]))
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
    ["t_i-1='JJ'", "t_i-2='DT'", "t_i-3='RB'", "t_i-2,t_i-1='DT','JJ'", "t_i-3,t_i-2,t_i-1='RB','DT','JJ'"]
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
