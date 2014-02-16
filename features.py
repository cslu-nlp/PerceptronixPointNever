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
# features.py: feature extractors


from __future__ import division

from string import digits

from decorators import Listify

PRE_SUF_MAX = 4
LEFT_PAD = ['<S1>', '<S0>']
RIGHT_PAD = ['</S0>', '</S1>']

# features for POS tagging (from Ratnaparkhi 1996, but ignoring the
# distinction between "rare" and "non-rare" words as in Collins 2002):
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
