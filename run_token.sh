export CONLL_PATH=~/uparse/data/nlp/treebank/treebank-2.0/combined/conll_token_scode
dist/Release/GNU-MacOSX/ai-parse -s optimize -p $CONLL_PATH -l 50 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl -o scode_token
