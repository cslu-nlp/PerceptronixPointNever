Perceptronix Point Never
========================

This is an implementation of part-of-speech tagging using the averaged 
percetron to train an HMM, as described in Collins 2002. As in Collins'
paper, the features used are the same as those used by Ratnaparkhi (1996). 
While this is unlikely to be fast (per se), every effort to optimize has 
been taken.

Usage
-----

    USAGE: ./ppn.py INPUT_ARGS OUTPUT_ARG1 [...] [-h] [-v]

    Input arguments (exactly one required):

        -i tag         train model on "token/tag"-format data in `tagged`
        -p pickled     read pickled training model from `pickled`

    Output arguments (at least one required):

        -D pickled     dump pickled training model to `sink`
        -E tagged      report accuracy on already-tagged data in `tagged`
        -T untagged    tag data in `untagged`, writing to stdout
    
    Optional arguments:

        -t t           number of training iterations (default: 10)
        -h             print this message and quit
        -v             increase verbosity

For anything else, UTSL...

License
-------

MIT License (BSD-like); see source.

Gotchas
-------

There is no guarantee that a pickled (and compressed) model file will be
compatible with a platform or Python version other than the one it was 
generated on.

What's with the name?
---------------------

It is an homage to experimental musician Daniel Lopatin, who performs under the name [Oneohtrix Point Never](pointnever.com).

Bugs, comments?
---------------

Contact [Kyle Gorman](mailto:gormanky@ohsu.edu).

References
----------

M. Collins. 2002. Discriminative training methods for hidden Markov models: Theory and experiments with perceptron algorithms. In _EMNLP_, 1-8.

A. Ratnaparkhi. 1996. A maximum entropy model for part-of-speech tagging. In _EMNLP_, 133-142.
