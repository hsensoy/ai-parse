ai-parse
========

Dependency Parsing by ai-lab.

```
$ dist/Release/GNU-MacOSX/ai-parse --help
Usage: ai-parse [options] [[--] args]

    -h, --help                show this help message and exit
    -o, --modelname=<str>     Model name
    -p, --path=<str>          CoNLL base directory including sections
    -s, --stage=<str>         [ optimize | train | parse ]
    -n, --maxnumit=<int>      Maximum number of iterations by perceptron. Default is 30
    -t, --training=<str>      Training sections for optimize and train. Apply sections for parse
    -d, --development=<str>   Development sections for optimize
    -e, --epattern=<str>      Embedding Patterns
    -l, --edimension=<int>    Embedding dimension
    -m, --maxrec=<int>        Maximum number of training instance
    -x, --etransform=<str>    Embedding Transformation
```

**ai-parse** command is a super-command to (defined by `-s` parameter)

*	_optimize_ (best hyper parameters based on dev set performance) a dependency parsing model using a training set, 
*	_train_ a dependency parsing model using a training set with given hyper parameters, and 
*	_parse_ a given a set of sentences by using  a given dependency parsing model.

`-o` parameter specifies a model name. String passed to the option is used as the model file name (`<string>.model`). For _optimize_ and _train_ stages this is the name of the file to be created after training is done, whereas this the model to be used by the parser when stage is _parse_

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

`-x` is used to define basis function to be used by the parser. Don't use this option today. This defined for future requirements.

### Two Sample Usage of ai-parse
Following call runs au-parse for optimising a dependency parser over a corpus in `~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_type_scode` root directory having 25 dimensional word embeddings. For arch scoring a combination of parent, child word embeddings, and distance between them is used (`-e` option). Model will be saved in to `scode_type.model` file once the run is complete.

```
$ cat run_type.sh 
export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_type_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 25 -e p0v_c0v_tl -o scode_type
```

Following call runs au-parse for optimising a dependency parser over a corpus in `~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_token_scode` root directory having 50 dimensional word embeddings. For arch scoring a combination of parent, parent left/right contexts, child, child left/right context word embeddings, and distance between them is used (`-e` option). Model will be saved in to `scode_token.model` file once the run is complete.

```
myhost:ai-parse husnusensoy$ cat run_token.sh 
export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_token_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 50 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl -o scode_token
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








	
