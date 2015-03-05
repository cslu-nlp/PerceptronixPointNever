Perceptronix Point Never
========================

Perceptronix Point Never (PPN) is an implementation of a part of speech
tagger using a hidden Markov model, the averaged perceptron classifier,
and a greedy decoding scheme. The classifier features are based loosely 
on those used by Ratnaparkhi 1996 and Collins 2002. Following Collins, 
the same features, including orthographic features, are used regardless 
of word frequency.

PPN has been tested on CPython 3.4 and PyPy3 (2.3.1, corresponding to 
Python 3.2); the latter is much, much faster. It requires three 
third-party packages: `nltk` and `jsonpickle` from PyPI and my own `nlup` 
library, available from GitHub; see `requirements.txt` for the versions 
used for testing.


Usage
-----

    usage: python -m PPN [-h] [-v] [-V] (-t TRAIN | -r READ)
                         (-u TAG | -w WRITE | -e EVALUATE) [-E EPOCHS] 
                         [-O ORDER]

    Perceptronix Point Never, by Kyle Gorman
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         enable verbose output
      -V, --really-verbose  even more verbose output
      -t TRAIN, --train TRAIN
                            training data
      -r READ, --read READ  read in serialized model
      -u TAG, --tag TAG     tag unlabeled data
      -w WRITE, --write WRITE
                            write out serialized model
      -e EVALUATE, --evaluate EVALUATE
                            evaluate on labeled data
      -E EPOCHS, --epochs EPOCHS
                            # of epochs (default: 10)
      -O ORDER, --order ORDER
                            Markov order (default: 2)


For anything else, UTSL.

License
-------

MIT License (BSD-like); see source.

What's with the name?
---------------------

It is an homage to experimental musician Daniel Lopatin, who performs 
under the name [Oneohtrix Point Never](http://pointnever.com).

Bugs, comments?
---------------

Contact [Kyle Gorman](mailto:gormanky@ohsu.edu).

References
----------

M. Collins. 2002. Discriminative training methods for hidden Markov models: Theory and experiments with perceptron algorithms. In _EMNLP_, 1-8.

A. Ratnaparkhi. 1996. A maximum entropy model for part-of-speech tagging. In _EMNLP_, 133-142.
