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




[^]: 3 is hard coded as a constant for today and may be parametrised in the future.



	
