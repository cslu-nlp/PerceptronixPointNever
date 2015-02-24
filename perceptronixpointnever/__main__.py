# Copyright (C) 2014-2015 Kyle Gorman
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


from nltk import tuple2str

from nlup.confusion import Confusion
from nlup.decorators import IO

from .tagger import Tagger, EPOCHS, ORDER, tagged_corpus, untagged_corpus

import logging

LOGGING_FMT = "%(message)s"


from argparse import ArgumentParser
argparser = ArgumentParser(prog="python -m perceptronixpointnever",
                    description="Perceptronix Point Never, by Kyle Gorman")
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="enable verbose output")
argparser.add_argument("-V", "--really-verbose", action="store_true",
                       help="even more verbose output")
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-t", "--train", help="training data")
input_group.add_argument("-r", "--read", help="read in serialized model")
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
# input
tagger = None
if args.read:
    logging.info("Reading pretrained tagger '{}'.".format(args.read))
    tagger = IO(Tagger.load)(args.read)
elif args.train:
    logging.info("Training on labeled data '{}'.".format(args.train))
    tagger = Tagger(order=args.order)
    tagger.fit(IO(tagged_corpus)(args.train), args.epochs)
# else unreachable
# output
if args.write:
    logging.info("Writing trained tagger to '{}'.".format(args.write))
    IO(tagger.dump)(args.write)
elif args.tag:
    logging.info("Tagging untagged data '{}'.".format(args.tag))
    for tokens in IO(untagged_corpus)(args.tag):
        print(str(tagger.tag(tokens)))
elif args.evaluate:
    logging.info("Evaluating tagged data '{}'.".format(args.evaluate))
    cx = Confusion()
    for sentence in IO(tagged_corpus)(args.evaluate):
        cx.batch_update(sentence.tags, tagger.tag(sentence.tokens).tags)
    print("Accuracy: {:.4f} [{:.4f}, {:.4f}].".format(cx.accuracy,
                                                      *cx.confint))
# else unreachable
