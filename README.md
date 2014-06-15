ai-parse
========
## Build
Currently ai-parse is tested on Mac OSX and Linux platforms. Makefile included with the bundle is platform aware. Use `help` to see supported Configurations

```
myhost:ai-parse husnusensoy$ make help
This makefile supports the following configurations:
    Debug Release Release-Linux 

and the following targets:
    build  (default target)
    clean
    clobber
    all
    help

Makefile Usage:
    make [CONF=<CONFIGURATION>] [SUB=no] build
    make [CONF=<CONFIGURATION>] [SUB=no] clean
    make [SUB=no] clobber
    make [SUB=no] all
    make help

Target 'build' will build a specific configuration and, unless 'SUB=no',
    also build subprojects.
Target 'clean' will clean a specific configuration and, unless 'SUB=no',
    also clean subprojects.
Target 'clobber' will remove all built files from all configurations and,
    unless 'SUB=no', also from subprojects.
Target 'all' will will build all configurations and, unless 'SUB=no',
    also build subprojects.
Target 'help' prints this message.
```
### Mac OS X
Use Debug/Release for Mac OS X platforms

```
make CONF=Release-icc
```
### Linux
Use Linux-Release for Mac OS X platforms

```
make CONF=Release-icc-Linux	 
```

## Options

Dependency Parsing by ai-lab.

Ensure that you run (or it is already in your .bashrc, .bash_profile,etc. files)

```
source $(INTEL_BASE)/bin/iccvars.sh intel64
```

It is most likely to be something like

```
source /opt/intel/bin/iccvars.sh intel64
```
```

$ dist/Release-icc/icc-MacOSX/ai-parse --help
2014-05-05 14:49:23 [INFO] (ai-parse.c:main:62) ai-parse v0.9.4 (Release)
Usage: ai-parse [options] [[--] args]

	-h, --help                show this help message and exit
    -v, --verbosity=<int>     Verbosity level. Minimum (Default) 0. Increasing values increase parser verbosity.
    -o, --modelname=<str>     Model name
    -p, --path=<str>          CoNLL base directory including sections
    -s, --stage=<str>         [ optimize | train | parse ]
    -n, --maxnumit=<int>      Maximum number of iterations by perceptron. Default is 50
    -t, --training=<str>      Training sections for optimize and train. Apply sections for parse
    -d, --development=<str>   Development sections for optimize
    -e, --epattern=<str>      Embedding Patterns
    -l, --edimension=<int>    Embedding dimension
    -m, --maxrec=<int>        Maximum number of training instance
    -x, --etransform=<str>    Embedding Transformation
    -k, --kernel=<str>        Kernel Type
    -a, --bias=<int>          Polynomial kernel additive term. Default is 1
    -c, --concurrency=<int>   Parallel MKL Slaves. Default is 90% of all machine cores
    -b, --degree=<int>        Degree of polynomial kernel. Default is 4
    -z, --lambda=<str>        Lambda multiplier for RBF Kernel.Default value is 0.025
    -u, --budget_type=<str>   Budget control methods. NONE|RANDOM
    -g, --budget_size=<int>   Budget Target for budget based perceptron algorithms. Default 50K

```

`-s` parameter defines the stage/mode of the parser. There are 3 valid stages/modes

*	_optimize_ (best hyper parameters based on dev set performance) a dependency parsing model using a training set
	* _optimize_ option generates a file with `.model` extension and model name (given by `-o` option). This file is used by _parse_ option to load the model.
*	_train_ a dependency parsing model using a training set with given hyper parameters, and 
*	_parse_ a given a set of sentences by using  a given dependency parsing model.
	* _parse_ option generates two files with `<model_name>.conll.gold` and `<model_name>.conll.model` format (`<model_name>` is given by `-o` option in `optimise` or `train` stage). Files are in 10 column CoNLL format with parent field difference. `<model_name>.conll.gold` includes true parent whereas `<model_name>.conll.model` includes predicted parent.	

`-v` parameter specifies the level of verbosity during parser execution. (This is not about logging)

* 0 is the minimum (default) verbosity level
* 1 causes hypothesis vectors to be dumped into a UNIX epoch labeled file after each perceptron pass over training data.

`-o` parameter specifies a model name. Refer `-s` option for more details

`-p` parameter refers to any CoNLL root directory. Expected directory structure is 

```
	<path>
		\---00
			\---wsj_00*.dp
		\---01
			\---wsj_01*.dp
		.
		.
		\---24
			\---wsj_24*.dp		
		
```

Refer to `scripts/enrich.py` to create clones of original ConLL directory enriched with embeddings.

`-n` is only relevant for _optimize_ and _train_ stages. For _optimize_ it is the maximum number of iterations in optimising dev set accuracy (Remember that _optimize_ can terminate before reaching this number. Another condition that will cause optimisation to stop is not to observe any accuracy improvement on development set in last 3 iterations[^].) 

`-t` is used to set sections to be used as the training set when _optimize_ and _train_ stages are concerned and test set to be parsed for _parse_ stage. Valid formats to be used with the option is 

* `<start>-<end>` to refer `[<start>, <end>)` section. For example, 2-22 refers all sections between 2 and 22 excluding section 22 but including section 2.
* `s1,s2,s3` to refer a list of sections defined as `{s1, s2,s3}`. For example 2,3,4 refers to sections 2,3, and 4.
* `s1` to refer a single section. For example 22 refers to section 22.

`-d` is used to set sections to be used as the development set when _optimize_ stage is used. Valid format for the option is the format used for `-t` option.

`-e` is used to define an embedding pattern to be used as an input in arc scoring (assign a score for a potential arc from a word (**p**arent) to another one (**c**hild) ) for Eisner's algorithm. Format of the parameter is `pattern1_pattern2_..._patternN` (a series of pattern strings concatenated by underscores). Valid pattern strings are:

* `p<N>v`: Refers to word embedding at position `i_p + <N>`. For example `p0v` refers to embedding for parent word itself, whereas `p-1v` refers to left context word of parent word given that `p != 1`, for which embedding is defined to be `0` vector. Just like left context (`p-1v`), right context (`p1v`) of a parent is a `0` vector given that parent word is the last word of the sentence.
* `c<N>v`: Refers to word embedding at position `i_c + <N>`. For example `c0v` refers to embedding for child word itself, whereas `c-1v` refers to left context word of child word given that `c != 1`, for which embedding is defined to be `0` vector. Just like left context (`c-1v`), right context (`c1v`) of a child is a `0` vector given that child word is the last word of the sentence.
* `tl`: Refers to _thresholded length_ defining 6 binary features for the number of words between parent and child (another way of saying absolute length between parent and child). Those are
	* `1 if length > 2 else 0`
	* `1 if length > 5 else 0`
	* `1 if length > 10 else 0`
	* `1 if length > 20 else 0`
	* `1 if length > 30 else 0`
	* `1 if length > 40 else 0`
* `rl`: Refers to _raw length_ defining the number of words between parent and child as a continuos value.

`-l` is used to set expected number of dimensions in word embeddings given in enriched CoNLL corpus. For example, this is 25 for majority of SCODE type based embeddings and 50 for majority of SCODE token based embeddings.

`-m` is used to restrict number of training instances to be used for _optimize_ and _train_. Only first `-m` instances will be used from the sections given by `-t` option. This option defined to perform experiments to see the effect of training instances used over development set accuracy.

`-x` is used to define basis function to be used by the parser. Don't use this option today. This defined for future requirements. Either `LINEAR` (default) or `QUADRATIC`.

`-c` is used to define concurrency to be used in parsing (Used by MKL routines. Intel auto parallelism uses 8-way parallelism on Linux 2-way parallelism on MacOS)

`-k` is used to define kernel function. Either `LINEAR` (default) or `POLYNOMIAL`.

`-a` is the bias parameter for `POLYNOMIAL` kernel.

`-b` is the degree of `POLYNOMIAL` kernel function.

`-z` is the lambda for `RBF` kernel.

`-u` is the budgeting type. Default is `NONE`. `RANDOM` uses a random pruning to reduce the number of hypothesis vectors.

`-g` is the budget limit for budget based perceptron algorithms. Default is `50K` hypothesis instances.

## Sample Runs
### Polynomial Kernel Perceptron Mode
#### Model Optimisation
Following call runs `ai-parse` to train a model called `kernel_limited` (this will create a model file called `kernel_limited.model` by termination) using `POLYNOMIAL` kernel and  executing parser in 16-way parallelism. Training will only use first `10000` sentences in `2-22` sections of corpus.

```
ai-parse -o kernel_limited -p $CONLL_ROOT -s optimize -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl -l 25 -k POLYNOMIAL -x LINEAR -c 16 -m 10000
```

#### Parsing
Following call runs `ai-parse` to apply kernel model in `kernel.model` file to parse `0,23,24` sections of corpus in `$CONLL_ROOT` base directory in 16-way parallelism. 

```
ai-parse -o kernel -p $CONLL_ROOT -s parse -t 0,23,24 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl -l 25 -x LINEAR -k POLYNOMIAL -c 16
```

### Feature Function Mode
#### Model Optimisation

Following call runs ai-parse for optimising a dependency parser over a corpus in `~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_type_scode` root directory having 25 dimensional word embeddings. For arch scoring a combination of parent, child word embeddings, and distance between them is used (`-e` option). Model will be saved in to `scode_type.model` file once the run is complete.

```
$ cat run_type.sh 
export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_type_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 25 -e p0v_c0v_tl -o scode_type -x QUADRATIC
```

Following call runs ai-parse for optimising a dependency parser over a corpus in `~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_token_scode` root directory having 50 dimensional word embeddings. For arch scoring a combination of parent, parent left/right contexts, child, child left/right context word embeddings, and distance between them is used (`-e` option). Model will be saved in to `scode_token.model` file once the run is complete.

```
myhost:ai-parse husnusensoy$ cat run_token.sh 
export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_token_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 50 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl -o scode_token -x QUADRATIC
```





[^]: 3 is hard coded as a constant for today and may be parametrised in the future.

ConLLCorpus Creation with Word Embeddings
==========================================
Before performing parsing experiments using `ai-parse` CoNLL corpus is expected to be enriched with word embeddings. 

We do this by adding a 11th column into standard ConLL file format for word embedding. Just like other attributes of a CoNLL record 11th column is separated by `<TAB>` character. Each dimension of embeddings are separated by a single space character.

Although `ai-parse` does not depend on process generating the replica of original CoNLL corpus with embeddings, `scripts/enrich_with_embeddings.py` provides a way of easily generating a CoNLL replica given an embeddings lookup file.

```
$ python scripts/enrich_with_embeddings.py --help
usage: enrich_with_embeddings.py [-h] [--delimiter DELIMITER]
                                 [--offset OFFSET] [--unk_as_default]
                                 [--token] [--unk_key UNK_KEY]
                                 [--length LENGTH]
                                 embeddings_file src target

Create a clone of CoNLL corpus by adding given embeddings

positional arguments:
  embeddings_file       Delimited embedding file first token to be word others
                        to be the embedding dimensions
  src                   Root directory of CoNLL corpus to be used as source
  target                Root directory of CoNLL corpus to be used as target

optional arguments:
  -h, --help            show this help message and exit
  --delimiter DELIMITER
                        File delimiter. Default is <TAB>
  --offset OFFSET       Starting offset of embedding token. Default: 1
  --unk_as_default      Use <unk> embedding as the default embedding.
                        Otherwise 0 embedding is used
  --token               Token based embeddings instead of type based
                        embeddings
  --unk_key UNK_KEY     UNK key to be used to replace unknown words
  --length LENGTH       Expected length of embeddings

python scripts/enrich_with_embeddings.py best-dis+om.enw.type.gz conll
conll_scode
```

### Options

``--delimiter`` is an option to define the dimension delimiter in `embeddings_file`.

``--offset`` is an option to define the starting offset of embeddings data. To be more precise given a row in `embeddings_file` delimited by `--delimiter`. First embedding dimension is pythonized by `row.split(--delimiter)[--offset]`.

``--unk_as_default`` is a boolean option to decide on embeddings of _unknown_ words. If option is set embedding for `--unk_key` in `embeddings_file` will be used. Otherwise a zero vector will be used.

``--token`` is another boolean option to decide `embeddings_file` format

* Option is set: This means that `embeddings_file` has token based format and there is a line for each line in ConLL corpus(order by section, filename). Any mismatch in word in `embeddings_file` line and corpus file line will cause an error.
* Option is not set: This means that `embeddings_file` has type based format and a Python dictionary will be constructed using the file and will be used to enrich CoNLL corpus.

`--unk_key` Refer to `--unk_as_default` option.

`--length` Expected dimension of embeddings vector in `embeddings_file`

### Positional Parameters

`embeddings_file` is the file (`.gz` is supported) either in _token_ or _type_ format. Refer to `--token` for more details.

`src` is the source CoNLL directory with 10 column corpus files. All subdirectories (section directories) will be read iteratively and enriched by given embeddings.

`target` is the target CoNLL directory to be created (if necessary ) by the script. All enriched sentences are written into this directory with section subdirectories.








	
