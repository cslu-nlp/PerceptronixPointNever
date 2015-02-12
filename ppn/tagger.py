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

from nlup import listify, tupleify, untagged_corpus, tagged_corpus,  \
                 Accuracy, JSONable, IO, SequenceAveragedPerceptron, \
                 TaggedSentence


EPOCHS = 10
ORDER = 2

DIGIT = "*DIGIT*"

PUNCTUATION = frozenset(punctuation)
NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve 
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty 
thirty fourty fifty sixty seventy eighty ninety hundred thousand million 
billion trillion quadrillion
zeros ones twos threes fours fives sixs sevens eights nines tens elevens 
twelves thirteens fourteens fifteens sixteens seventeens eighteens 
nineteens twenties thirties fourties fifties sixties seventies eighties 
nineties hundreds thousands millions billions trillions quadrillions
""".upper().split())



# feature extractors


@lru_cache(128)
def fstring(key, value):
    return "{}='{}'".format(key, value)


def isnumberlike(token):
    """
    Feature incporating a relatively broad definition of "numberhood"
    """
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
    if token in NUMBER_WORDS or all(part in NUMBER_WORDS for part in
                                    token.split("-")):
        return True
    return False


@tupleify
def efeats_now(token, ftoken):
    """
    Emission features associated with the current observation
    """
    yield fstring("w_i", ftoken)
    # no other features will match
    if ftoken == DIGIT:
        return
    else:
        # casing
        if token.islower():
            yield "*lowercase*"
        elif token.isupper():
            yield "*uppercase*"
        elif token.istitle():
            yield "*titlecase*"
        # punctuation
        if all(char in PUNCTUATION for char in ftoken):
            yield "*punctuation*"
        # numberlikeness
        elif isnumberlike(ftoken):
            yield "*numberlike*"
        else:
            # word-like
            if "-" in ftoken:
                yield "*hyphenated*"
            #if "'" in token:
            #   yield "*apostrophe*"
            yield fstring("pre1(w_i)", ftoken[0])
            yield fstring("suf3(w_i)", ftoken[-3:])


@tupleify
def efeats_earlier(ftokens, i):
    """
    Emission features associated with earlier observations
    """
    # first token
    if i == 0:
        yield "*first-token*"
        return
    # otherwise
    yield fstring("w_i-1", ftokens[i - 1])
    if ftokens[i - 1] != DIGIT:
        yield fstring("suf3(w_i-1)", ftokens[i - 1][-3:])
    # second token
    if i == 1:
        yield "*second-token*"
    else:
        yield fstring("w_i-2", ftokens[i - 2])


@tupleify
def efeats_later(ftokens, i):
    """ 
    Emission features associated with later observations
    """
    # last word
    if i == len(ftokens) - 1:
        yield "*ultimate-token*"
        return
    # otherwise
    yield fstring("w_i+1", ftokens[i + 1])
    if ftokens[i + 1] != DIGIT:
        yield fstring("suf3(w_i+1)", ftokens[i + 1][-3:])
    # penultimate
    if i == len(ftokens) - 2:
        yield "*penultimate-token*"
    else:
        yield fstring("w_i+2", ftokens[i + 2])


@tupleify
def efeats(tokens):
    """
    Compute list of lists of emission features for each token in 
    the `tokens` iterator
    """
    utokens = [DIGIT if token.isdigit() else token for token in tokens]
    ftokens = [token.upper() for token in tokens]
    for (i, (utoken, ftoken)) in enumerate(zip(utokens, ftokens)):
        yield ("*bias*",) + efeats_now(utoken, ftoken) + \
              efeats_earlier(ftokens, i) + efeats_later(ftokens, i)

@tupleify
def tfeats(tags):
    """
    Compute a list of features for a single token using an iterator of
    tags; this also partially determines the Markov order. An example:

    >>> d = 3
    >>> tags = "RB DT JJ NN".split()
    >>> tfeats([])
    []
    >>> tfeats(tags[:1])
    ["t_i-1='RB'"]
    >>> tfeats(tags[:2])
    ["t_i-1='DT'", "t_i-2='RB'", "t_i-2,t_i-1='RB','DT'"]
    >>> tfeats(tags[:3])
    ["t_i-1='JJ'", "t_i-2='DT'", "t_i-2,t_i-1='DT','JJ'", "t_i-3='RB'", "t_i-3,t_i-2,t_i-1='RB','DT','JJ'"]
    """
    feats = []
    if not tags:
        return feats
    i = 1
    tfeat_key = "t_i-{}".format(i)
    feats.append(fstring(tfeat_key, tags[-i]))
    for i in range(2, 1 + len(tags)):
        feats.append(fstring("t_i-{}".format(i), tags[-i]))
        tfeat_key = "t_i-{},{}".format(i, tfeat_key)
        vstring = ",".join("'{}'".format(tag) for tag in tags[-i:])
        feats.append("{}={}".format(tfeat_key, vstring))
    return feats


class Tagger(JSONable):

    """
    Part-of-speech tagger, backed by a classifier
    """

    def __init__(self, efeats_fnc=efeats, tfeats_fnc=tfeats, order=ORDER):
        self.tagger = SequenceAveragedPerceptron(efeats_fnc, tfeats_fnc,
                                                 order)

    def fit(self, sentences, epochs=EPOCHS):
        sentences = list(sentences)
        YY = []
        XX = []
        for sentence in sentences:
            YY.append(sentence.tags)
            XX.append(sentence.tokens)
        self.tagger.fit(YY, XX, epochs)

    def tag(self, tokens):
        return TaggedSentence(tokens, self.tagger.predict(tokens))

    @listify
    def batch_tag(self, tokens_list):
        for tokens in tokens_list:
            yield self.tag(tokens)
