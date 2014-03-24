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
LEFT_PAD = ["<S1>", "<S0>"]
RIGHT_PAD = ["</S0>", "</S1>"]

class FeatureExtractorI(object):
    """
    Interface for extracting features for tagging or chunking using
    an HMM.

    Subclasses must define:
    ``extract_sent_efs()``: "emission" features, those which can be 
                             precomputed offline
    ``extract_sent_efs()``: "transition" features, those which depend on
                             labels of other tokens and which are 
                             computed online

    As an optimization, both concrete instantiations of this class 
    also implement `bigram_tf()` (which generates a bigram transition
    feature string ) and `trigram_tf()` (which genereates a trigram 
    transition feature string).
    """

    def extract_sent_efs(tokens):
        """
        Create a list of emission features for each token. There is 
        one list of emission feature strings per token. Tokens are 
        encoded as a list(str).

        :type tokens: list(str)
        :param gold: List of word tokens.
        :rtype: list(list(str))
        """
        raise NotImplementedError

    def extract_sent_tfs(tags):
        """
        Return a list of transition features for each tag. There is 
        one list of transition feature strings per tag. Tags are 
        encoded as a list(str).

        :type tags: list(str)
        :param tags: List of part-of-speech tags.
        :rtype: list(list(str))
        """
        raise NotImplementedError

class TaggerFeatureExtractor(FeatureExtractorI):

    """
    Extracts part-of-speech tagging features used by Ratnaparkhi (1998)
    and Collins (2002) in their tagging experiments.

    Features for POS tagging:

    b: bias
    w-2=X: two tokens back
    w-1=X: previous token
    w=X: current token
    w+1=X: next token
    w+2=X: two tokens ahead
    p1=X: first character
    p2=X: first two characters
    p3=X: first three characters
    p4=X: first four characters
    s1=X: last character
    s2=X: last two characters
    s3=X: last three characters
    s4=X: last four characters
    h: contains a hyphen?
    n: contains a number?
    u: contains an uppercase character
    t-1=X: previous tag
    t-2=X,t-1=X: previous two tags
    """

    @staticmethod
    @Listify
    def extract_sent_efs(tokens):
        """
        Create a list of Ratnaparkhi/Collins emission features for each
        token. There is one list of emission feature strings per token. 
        Tokens are encoded as a list(str).

        :type tokens: list(str)
        :param tokens: List of part-of-speech tokens.
        :rtype: list(list(str))
        """
        padded_tokens = LEFT_PAD + [t.lower() for t in tokens] + RIGHT_PAD
        for (i, ftoken) in enumerate(padded_tokens[2:-2]):
            # even though `ftoken` is the current token, `i` is the index 
            # of two tokens back
            featset = ['b']  # initialize with bias term
            # tokens nearby
            featset.append("w-2='{}'".format(padded_tokens[i]))
            featset.append("w-1='{}'".format(padded_tokens[i + 1]))
            featset.append("w='{}'".format(ftoken)) 
            featset.append("w+1='{}'".format(padded_tokens[i + 3]))
            featset.append("w+2='{}'".format(padded_tokens[i + 4]))
            for j in xrange(1, 1 + min(len(ftoken), PRE_SUF_MAX)):
                featset.append("p({})='{}'".format(j, ftoken[:+j]))  # pre
                featset.append("s({})='{}'".format(j, ftoken[-j:]))  # suf
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

    @staticmethod
    @Listify
    def extract_sent_tfs(tags):
        """
        Create a list of Ratnaparkhi/Collins transition features for each
        tag. There is one list of transition feature strings per tag. Tags
        are encoded as a list(str).

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
            yield TaggerFeatureExtractor.extract_token_tfs(tags[i], \
                                                            tags[i + 1])

    # functions to generate those tag features
    @staticmethod
    @Memoize
    def extract_token_tfs(prev_prev_tag, prev_tag):
        """
        Create a list of Ratnaparkhi/Collins transition features for
        a single tag. Both arguments are strings.

        :type prev_prev_tag: str
        :param prev_prev_tag: the tag two tags back
        :type prev_tag: str
        :param prev_tag: the previous tag
        :rtype: list(str)
        """
        bigram_tf_string = TaggerFeatureExtractor.bigram_tf(prev_tag)
        trigram_tf_string = TaggerFeatureExtractor.trigram_tf(
                                                   prev_prev_tag, 
                                                   bigram_tf_string)
        return [bigram_tf_string, trigram_tf_string]

    @staticmethod
    def bigram_tf(prev_tag):
        """
        Create bigram feature string
        
        :type prev_tag: str
        :param prev_tag: the previous tag
        :rtype: str 
        """
        return "t-1='{}'".format(prev_tag)

    @staticmethod
    def trigram_tf(prev_prev_tag, bigram_feature_string):
        """
        Create trigram feature string

        :type prev_prev_tag: str
        :param prev_prev_tag: the tag two tags back
        :type bigram_feature_string: str
        :param bigram_feature_string: a bigram feature string for the
                                      previous tag, created with bigram_tf
        """
        return "t-2='{}',{}".format(prev_prev_tag, bigram_feature_string)


class ChunkerFeatureExtractor(FeatureExtractorI):
    """
    Extracts chunking features used by Ratnaparkhi (1998) and Collins 
    (2002) in their chunking experiments.

    Features for chunking:

    b: bias (omnipresent)
    w-1=X: previous token
    w-2=X: two tokens back
    w=X: current token
    w+1=X: next token
    w+2=X: two tokens ahead
    w-2,w-1=X,Y: previous two tokens
    w-1,w=X,Y: previous token and current token
    w,w+1=X,Y: current token and next token
    w+1,w+2=X,Y: next two tokens
    t-1=X: previous POS tag
    t-2=X: two POS tags back
    t=X: current POS tag
    t+1=X: next POS tag
    t+2=X: two POS tags ahead
    t-2,t-1=X,Y: previous two POS tags
    t-1,t=X,Y: previous POS tag and current POS tag
    t+1,t+2=X,Y: next two POS tags
    t-2,t-1,t=X,Y,Z: previous two POS tags and current POS tag
    t-1,t,t+1=X,Y,Z: previous POS tag, current POS tag, and next POS tag
    t,t+1,t+2=X,Y,Z: current POS tag and next two POS tags
    """
    @staticmethod
    @Listify
    def extract_sent_efs(tokens):
        """
        Create a list of Ratnaparkhi/Collins emission features for each
        token. There is one list of emission feature strings per token. 
        Tokens are encoded as a list(str).

        :type tokens: list(str)
        :param tokens: List of part-of-speech tokens.
        :rtype: list(list(str))
        """
        padded_tokens = LEFT_PAD + [t.lower() for t in tokens] + RIGHT_PAD
        for (i, ftoken) in enumerate(padded_tokens[2:-2]):
            # even though `ftoken` is the current token, `i` is the index 
            # of two tokens back
            featset = ['b']  # initialize with bias term
            # tokens nearby
            featset.append("w-2='{}'".format(padded_tokens[i]))
            featset.append("w-1='{}'".format(padded_tokens[i + 1]))
            featset.append("w='{}'".format(ftoken)) 
            featset.append("w+1='{}'".format(padded_tokens[i + 3]))
            featset.append("w+2='{}'".format(padded_tokens[i + 4]))
            featset.append("w-2='{}',w-1='{}'".format(padded_tokens[i],
                                               padded_tokens[i + 1]))
            featset.append("w-1='{}',w='{}'".format(padded_tokens[i + 1],
                                               padded_tokens[i + 2]))
            featset.append("w='{}',w+1='{}'".format(ftoken,
                                               padded_tokens[i + 3]))
            featset.append("w+1='{}',w+2='{}'".format(padded_tokens[i + 3],
                                               padded_tokens[i + 4]))
        # FIXME no POS tag-related features, yet
        yield featset


if __name__ == '__main__':
    import doctest
    doctest.testmod()
