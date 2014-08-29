#!/usr/bin/env python -O
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
# sequencetagger.py: online sequence tagging library


import logging

from time import time
from random import Random
from heapq import nlargest
from functools import partial
from operator import itemgetter
from collections import defaultdict, namedtuple

from jsonable import JSONable
from confusion import Accuracy
from decorators import reversify


INF = float("inf")


class Perceptron(JSONable):

    """
    The multiclass perceptron with sparse binary feature vectors:

    Each class (i.e., label, outcome) is represented as a hashable item,
    such as a string. Features are represented as hashable objects
    (preferably strings, as Python dictionaries have been aggressively
    optimized for this case). Presence of a feature indicates that that
    feature is "firing" and absence indicates that that it is not firing.
    This class is primarily to be used as an abstract base class; in most
    cases, the regularization and stability afforded by the averaged
    perceptron (`AveragedPerceptron`) or the passive-aggressive classifier
    (`PAClassifier`) will be worth it.

    The perceptron was first proposed in the following paper:

    F. Rosenblatt. 1958. The perceptron: A probabilistic model for
    information storage and organization in the brain. Psychological
    Review 65(6): 386-408.
    """

    # constructor

    def __init__(self, *, default=None, seed=None):
        self.classes = {default}
        self.random = Random(seed)
        self.weights = defaultdict(partial(defaultdict, int))

    # scoring

    def score(self, x, y):
        """
        Get score for one class (`y`) according to the feature vector `x`
        """
        return sum(self.weights[feature][y] for feature in x)

    def scores(self, x):
        """
        Get scores for all classes according to the feature vector `x`
        """
        scores = dict.fromkeys(self.classes, 0)
        for feature in x:
            for (cls, weight) in self.weights[feature].items():
                scores[cls] += weight
        return scores

    # prediction

    def predict_slow(self, x):
        """
        Get scores for all classes using the feature vector `x`. If/when
        ties arise, resolve them randomly. This is slower than `predict`.
        """
        scores = self.scores(x)
        (_, max_score) = max(scores.items(), key=itemgetter(1))
        argmax_scores = [cls for (cls, score) in scores.items()
                         if score == max_score]
        return self.random.choice(argmax_scores)

    def predict(self, x):
        """
        Same as `predict_slow`, but ties are resolved according to
        dictionary order, rather than randomly. Obviously, this is faster
        than `predict_slow`.
        """
        scores = self.scores(x)
        (argmax_score, _) = max(scores.items(), key=itemgetter(1))
        return argmax_score

    def fit_one(self, x, y):
        self.classes.add(y)
        yhat = self.predict(x)
        if y != yhat:
            self.update(x, y, yhat)
        return yhat

    def fit(self, X, Y, T=1):
        data = list(zip(X, Y))  # which is a copy
        logging.info("Starting {} epochs of training.".format(T))
        for i in range(1, 1 + T):
            logging.info("Starting epoch {:>2}.".format(i))
            tic = time()
            accuracy = Accuracy()
            self.random.shuffle(data)
            for (x, y) in data:
                yhat = self.fit_one(x, y)
                accuracy.update(y, yhat)
            logging.debug("Epoch {:>2} accuracy.:\t{:.04f}.".format(i,
                                                   accuracy.accuracy))
            logging.debug("Epoch {:>2} time:\t{}s.".format(i,
                                          int(time() - tic)))

    def update(self, x, y, yhat, tau=1):
        """
        Given feature vector `x`, reward correct observation `y` and
        punish incorrect hypothesis `yhat` with the update `tau`
        """
        for feature in x:
            feature_ptr = self.weights[feature]
            feature_ptr[y] += tau
            feature_ptr[yhat] -= tau


TrellisCell = namedtuple("TrellisCell", ["score", "pointer"])


class SequencePerceptron(Perceptron):
    """
    Perceptron with Viterbi-decoding powers
    """

    def __init__(self, *, tfeats_fnc=None, O=0, **kwargs):
        super(SequencePerceptron, self).__init__(**kwargs)
        self.tfeats_fnc = tfeats_fnc
        self.O = O

    def predict(self, xx):
        """
        Tag a sequence by applying the Viterbi algorithm:

        1. Compute tag-given-token forward probabilities and backtraces
        2. Compute the most probable final state
        3. Follow backtraces from this most probable state to generate
           the most probable tag sequence.

        The time complexity of this operation is O(n t^2) where n is the
        sequence length and t is the cardinality of the tagset.
        """
        trellis = self._trellis(xx)
        (best_last_state, _) = max(trellis[-1].items(), key=itemgetter(1))
        return self._traceback(trellis, best_last_state)

    def _trellis(self, xx):
        """
        Construct a trellis for decoding. The trellis is represented as a
        list, in which each element represents a single point in time.
        These elements are dictionaries mapping from state labels to
        `TrellisCell` elements, which contain the state score and a
        backpointer.
        """
        if self.O <= 0:
            return self._markov0_trellis(xx)
        else:
            return self._viterbi_trellis(xx)

    def _markov0_trellis(self, xx):
        """
        Construct a trellis for Markov order-0.
        """
        trellis = [{state: TrellisCell(score, None) for (state, score) in \
                    self.scores(xx[0]).items()}]
        for x in xx[1:]:
            (pstate, (pscore, _)) = max(trellis[-1].items(),
                                        key=itemgetter(1))
            column = {state: TrellisCell(pscore + escore, pstate) for \
                      (state, escore) in self.scores(x).items()}
            trellis.append(column)
        return trellis

    def _viterbi_trellis(self, xx):
        """
        Construct the trellis for Viterbi decoding assuming a non-zero
        Markov order.
        """
        # first case is special
        trellis = [{state: TrellisCell(score, None) for (state, score) in
                    self.scores(xx[0]).items()}]
        for x in xx[1:]:
            pcolumns = trellis[-self.O:]
            # store previous state scores
            pscores = {state: score for (state, (score, pointer)) in \
                       pcolumns[-1].items()}
            # store best previous state + transmission scores
            ptscores = {state: TrellisCell(-INF, None) for state in \
                        self.classes}
            # find the previous state which maximizes the previous state +
            # the transmission scores
            for (pstate, pscore) in pscores.items():
                tfeats = self.tfeats_fnc(self._traceback(pcolumns, pstate))
                for (state, tscore) in self.scores(tfeats).items():
                    ptscore = pscore + tscore
                    (best_ptscore, _) = ptscores[state]
                    if ptscore > best_ptscore:
                        ptscores[state] = TrellisCell(ptscore, pstate)
            # combine emission, previous state, and transmission scores
            column = {}
            for (state, escore) in self.scores(x).items():
                (ptscore, pstate) = ptscores[state]
                column[state] = TrellisCell(ptscore + escore, pstate)
            trellis.append(column)
        return trellis

    @reversify
    def _traceback(self, trellis, state):
        for column in reversed(trellis):
            yield state
            state = column[state].pointer

    def fit_one(self, xx, yy):
        self.classes.update(yy)
        # decode to get predicted sequence
        yyhat = self.predict(xx)
        for (i, (x, y, yhat)) in enumerate(zip(xx, yy, yyhat)):
            if y != yhat:
                # add hypothesized t-features to observed e-features
                x += self.tfeats_fnc(yyhat[i - self.O:i])
                self.update(x, y, yhat)
        return yyhat

    def fit(self, XX, YY, T=1):
        data = list(zip(XX, YY))
        logging.info("Starting {} epochs of training.".format(T))
        for i in range(1, 1 + T):
            logging.info("Starting epoch {:>2}.".format(i))
            tic = time()
            accuracy = Accuracy()
            self.random.shuffle(data)
            for (xx, yy) in data:
                yyhat = self.fit_one(xx, yy)
                for (y, yhat) in zip(yy, yyhat):
                    accuracy.update(y, yhat)
            logging.debug("Epoch {:>2} accuracy:\t{:.04f}.".format(i,
                                                   accuracy.accuracy))
            logging.debug("Epoch {:>2} time elapsed:\t{}s.".format(i,
                                                      int(time() - tic)))


class LazyWeight(JSONable):

    """
    Helper class for `AveragedPerceptron`:

    Instances of this class are essentially triplets of values which
    represent a weight of a single feature in an averaged perceptron.
    This representation permits "averaging" to be done implicitly, and
    allows us to take advantage of sparsity in the feature space.
    First, as the name suggests, the `summed_weight` variable is lazily
    evaluated (i.e., computed only when needed). This summed weight is the
    one used in actual inference: we need not average explicitly. Lazy
    evaluation requires us to store two other numbers. First, we store the
    current weight, and the last time this weight was updated. When we
    need the real value of the summed weight (for inference), we "freshen"
    the summed weight by adding to it the product of the real weight and
    the time elapsed.

    # initialize
    >>> t = 0
    >>> lw = LazyWeight(t=t)
    >>> t += 1
    >>> lw.update(t, 1)
    >>> t += 1
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

    def __init__(self, default_factory=int, t=0):
        self.timestamp = t
        self.weight = default_factory()
        self.summed_weight = default_factory()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__dict__)

    def _freshen(self, t):
        """
        Apply queued updates, and update the timestamp
        """
        self.summed_weight += (t - self.timestamp) * self.weight
        self.timestamp = t

    def get(self, t):
        """
        Return an up-to-date sum of weights
        """
        self._freshen(t)
        return self.summed_weight

    def update(self, value, t):
        """
        Bring sum of weights up to date, then add `value` to the weight
        """
        self._freshen(t)
        self.weight += value


class AveragedPerceptron(Perceptron):
    """
    The multiclass perceptron with sparse binary feature vectors, with
    averaging for stability and regularization.

    Averaging was originally proposed in the following paper:

    Y. Freund and R.E. Schapire. 1999. Large margin classification using
    the perceptron algorithm. Machine Learning 37(3): 227-296.
    """

    def __init__(self, *, default=None, seed=None):
        self.classes = {default}
        self.random = Random(seed)
        self.weights = defaultdict(partial(defaultdict, LazyWeight))
        self.time = 0

    def score(self, x, y):
        """
        Get score for one class (`y`) according to the feature vector `x`
        """
        return sum(self.weights[feature][y].get(self.time) for
                   feature in x)

    def scores(self, x):
        """
        Get scores for all classes according to the feature vector `x`
        """
        scores = dict.fromkeys(self.classes, 0)
        for feature in x:
            for (cls, weight) in self.weights[feature].items():
                scores[cls] += weight.get(self.time)
        return scores

    def fit_one(self, x, y):
        retval = super(AveragedPerceptron, self).fit_one(x, y)
        self.time += 1
        return retval

    def update(self, x, y, yhat, tau=1):
        for feature in x:
            feature_ptr = self.weights[feature]
            feature_ptr[y].update(+tau, self.time)
            feature_ptr[yhat].update(-tau, self.time)


class SequenceAveragedPerceptron(AveragedPerceptron, SequencePerceptron):

    def __init__(self, *, tfeats_fnc=None, O=0, **kwargs):
        super(SequenceAveragedPerceptron, self).__init__(**kwargs)
        self.tfeats_fnc = tfeats_fnc
        self.O = O


class PAClassifier(Perceptron):
    """
    The multiclass perceptron with sparse binary feature vectors, using
    a hinge loss function to enforce a unit margin.

    This algorithm was proposed in the following paper:

    K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shvartz, and Y. Singer.
    2006. Online passive-aggressive algorithms. Journal of Machine
    Learning Research. 7: 551-585.
    
    Strictly speaking, this is a variant known as "PA-I", which is more
    tolerant to noise. However, the default value for C, the aggressiveness
    parameter, is infinite. In this case, this is equivalent to the
    original "PA" algorithm.
    """

    def __init__(self, *, default=None, seed=None, C=INF):
        self.classes = {default}
        self.random = Random(seed)
        self.C = C
        self.weights = defaultdict(partial(defaultdict, int))

    def tau(self, x, margin):
        return min(self.C, (1. - margin) / (2 * len(x)))

    def fit_one(self, x, y):
        self.classes.add(y)
        scores = self.scores(x)
        (yhat, margin) = PAClassifier._fit_one_helper(y, self.scores(x))
        if margin < 1.:
            self.update(x, y, yhat, self.tau(x, margin))
        return yhat

    @staticmethod
    def _fit_one_helper(y, scores):
        (top, second) = nlargest(2, scores, key=itemgetter(1))
        (top_cls, top_score) = top
        (_, second_score) = second
        if top_cls == y:
            # highest ranked class is y, and next best class is yhat
            yhat = y
            margin = top_score - second_score
        else:
            # top class is yhat, so we look up the score for y
            yhat = top_cls
            margin = scores[y] - top_cls
        return (yhat, margin)


# TODO: do sequence classification with PAClassifier


if __name__ == "__main__":
    import doctest
    doctest.testmod()
