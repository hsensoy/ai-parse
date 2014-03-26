export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_type_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 25 -e p0v_c0v_tl -o scode_type
