Perceptronix Point Never
========================

This is an implementation of an HMM sequence tagger trained using the 
averaged perceptron algorithm. Features for POS tagging and chunking are 
as described in Ratnaparkhi 1996 and Collins 2002. Following Collins, the 
same features, including orthographic features, are used regardless of 
word frequency.

Usage
-----

Perceptronix Point Never 0.8, by Kyle Gorman and Steven Bedrick

    USAGE: ./ppn.py [-i|-p] file [-I|-P|-E] file [-T 10] [-c] [-h] [-v]

    Input arguments (at least one required):

        -i file        train on labeled data
        -p file        read in serialized model

    Output arguments (at least one required):

        -I file        tag or chunk unlabeled data `unlabeled
        -E file        evaluate on labeled data
        -P file        write out serialized model
    
    Optional arguments:

        -T iters       number of training iterations       [default: 10]
        -c             run as chunker, not as a tagger     [default: no]

        -h             print this message and quit
        -v             increase verbosity

    All inputs should be whitespace-delimited with one sentence per line.

    Tagger training/evaluation: "token/POS-tag"
    Tagging: bare tokens (no POS tags)

    Chunker training/evaluation: "token/POS-tag/chunk-tag"
    Chunking: "token/POS-tag" tokens (no chunk tags)

For anything else, UTSL.

A few Python-based scripts have been added to assist in tagging: 

    * `universal` converts Penn Treebank tags to the Petrov et al. (2012) universal tagset; their results suggest that you may still want to train with a full tagset, and then convert to the universal tagset for downstream applications (or evaluation)
    * `confusion` creates a tagging confusion matrix in CSV format
    * `untag` removes tags from data for performing experiments
    * `tree2pos` converts PTB-style trees to token/tag format, ignoring non-(pre)terminals; this is useful 

Requirements
------------

Three 3rd-party Python modules are required:

* `numpy` for numeric arrays
* `jsonpickle` for model (de)serialization
* `nltk` for interfaces and evaluation code

License
-------

MIT License (BSD-like); see source.

What's with the name?
---------------------

It is an homage to experimental musician Daniel Lopatin performing under the name [Oneohtrix Point Never](pointnever.com).

Bugs, comments?
---------------

Contact [Kyle Gorman](mailto:gormanky@ohsu.edu).

References
----------

M. Collins. 2002. Discriminative training methods for hidden Markov models: Theory and experiments with perceptron algorithms. In _EMNLP_, 1-8.

A. Ratnaparkhi. 1996. A maximum entropy model for part-of-speech tagging. In _EMNLP_, 133-142.

S. Petrov, D. Das, and R. McDonald. 2012. A universal part-of-speech tagset. In _LREC_, 2089-2096.
