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
# features.py: feature extractors


from __future__ import division

from string import digits

from decorators import Listify, Memoize

PRE_SUF_MAX = 4
LEFT_PAD = ['<S1>', '<S0>']
RIGHT_PAD = ['</S0>', '</S1>']

# features for POS tagging (from Ratnaparkhi 1996, but ignoring the
# distinction between "rare" and "non-rare" words as in Collins 2002):
#
# w-2=X: two tokens back
# w-1=X: previous token
# w=X: current token
# w+1=X: next token
# w+2=X: two tokens ahead
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
# t-1=X: previous tag
# t-2=X,t-1=X: previous two tags


@Listify
def extract_sent_efs(tokens):
    """
    Given a list of tokens, extract all emission-related features in the
    form of a list(list(str)) where each inner list contains a list of
    (non-null) features for the corresponding token
    """
    padded_tokens = LEFT_PAD + [t.lower() for t in tokens] + RIGHT_PAD
    for (i, ftoken) in enumerate(padded_tokens[2:-2]):
        # even though `ftoken` is the current token, `i` is the index of
        # two tokens back
        featset = ['b']  # initialize with bias term
        # tokens nearby
        featset.append('w-2="{}"'.format(padded_tokens[i]))
        featset.append('w-1="{}"'.format(padded_tokens[i + 1]))
        featset.append('w="{}"'.format(ftoken))  # == padded_tokens[i + 2]
        featset.append('w+1="{}"'.format(padded_tokens[i + 3]))
        featset.append('w+2="{}"'.format(padded_tokens[i + 4]))
        for j in xrange(1, 1 + min(len(ftoken), PRE_SUF_MAX)):
            featset.append('p({})="{}"'.format(j, ftoken[:+j]))  # prefix
            featset.append('s({})="{}"'.format(j, ftoken[-j:]))  # suffix
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


@Listify
def extract_sent_tfs(tags):
    """
    Given a list of tags, extract trigram tag features
    """
    # for the first two tokens, there are no tag features; these would be
    # identical to the word features anyways
    for _ in xrange(2):
        yield []
    # general case
    for i in xrange(len(tags) - 2):
        yield extract_token_tfs(tags[i], tags[i + 1])

# functions to generate those tag features


def bigram_tf(prev_tag):
    return 't-1="{}"'.format(prev_tag)


def trigram_tf(prev_prev_tag, bigram_feature_string):
    return 't-2="{}",{}'.format(prev_prev_tag, bigram_feature_string)


@Memoize
@Listify
def extract_token_tfs(prev_prev_tag=None, prev_tag=None):
    if not prev_tag:   # no tag history
        return
    bigram_tf_string = bigram_tf(prev_tag)
    yield bigram_tf_string
    if prev_prev_tag:  # at least two tags of history
        yield trigram_tf(prev_prev_tag, bigram_tf_string)

# TODO NP-chunking features (from Collins 2002 via Ratnaparkhi 1996):
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
# t-1=X: previous POS tag
# t-2=X: two POS tags back
# t=X: current POS tag
# t+1=X: next POS tag
# t+2=X: two POS tags ahead
#
# t-2,t-1=X,Y: previous two POS tags
# t-1,t=X,Y: previous POS tag and current POS tag
# t+1,t+2=X,Y: next two POS tags
#
# t-2,t-1,t=X,Y,Z: previous two POS tags and current POS tag
# t-1,t,t+1=X,Y,Z: previous POS tag, current POS tag, and next POS tag
# t,t+1,t+2=X,Y,Z: current POS tag and next two POS tags
