//
//  corpus.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 13/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_corpus_h
#define Perceptron_GLM_NLP_Tasks_corpus_h
#include "darray.h"
#include "hashmap.h"
#include "datastructure.h"
#include "vector.h"


#define EXAMPLE_CONLL_DIR "/Users/husnusensoy/uparse/data/nlp/treebank/treebank-2.0/combined/conll"

#define  STOP  "<STOP>"
#define START  "*"
//static const char* ROOT = "root";


#define IS_ARC_VALID(from,to, length) check((from) != (to) && (from) <= (length) && (from) >= 0 && (to)>= 1 && (to) <= (length), "Arc between suspicious words %d to %d for sentence length %d", (from), (to), (length))


#define MAX_SENT_LENGTH 20

struct Word {
    int id;
    int parent;
    
    int predicted_parent;       // Parent predicted by the model.
    
    char *form;
    char *postag;
    
    DArray *conll_piece;

    vector embedding;
};

typedef struct Word* Word;

//Word parse_word( char* line, bool read_vector );



FeatureMatrix FeatureMatrix_create(int sent_length, uint32_t embedding_length, bool has_discrete_features);



enum EmbeddingTranformation{
    QUADRATIC,
    LINEAR
};

struct CoNLLCorpus {
    const char *base_dir;
    DArray* sections;

    DArray *sentences;

    bool hasembeddings;
    DArray *disrete_patterns_parts;

    Word Root;
    size_t word_embedding_dimension;
    size_t transformed_embedding_length;
};

typedef struct CoNLLCorpus* CoNLLCorpus;


struct EmbeddingPattern {
    int offset;
    char node;
    char subnode;
};

typedef struct EmbeddingPattern* EmbeddingPattern;



CoNLLCorpus create_CoNLLCorpus(const char* base_dir, DArray *sections, int embedding_dimension, DArray* discrete_patterns) ;
void read_corpus(CoNLLCorpus coprus, bool build_feature_matrix);

void free_CoNLLCorpus(CoNLLCorpus corpus, bool free_feature_matrix);



void add_word(FeaturedSentence sentence, Word word);

FeaturedSentence FeatureSentence_create();
void FeatureSentence_free(FeaturedSentence sent, bool free_words);

/**
 * @brief Constructs feature_matrix for a given sentence
 * @param featuremap String: Integer map for discrete features. 
 * @param corpus CoNLLCorpus object.
 * @param sentence_idx Sentence for which feature matrix is built.
 */
void set_FeatureMatrix(Hashmap* featuremap, CoNLLCorpus corpus, int sentence_idx);
void free_featureMatrix(FeatureMatrix matrix);


void free_feature_matrix(CoNLLCorpus corpus, int sentence_idx);

void build_adjacency_matrix(CoNLLCorpus corpus, int sentence_idx, vector embeddings_w, vector discrete_w);
void set_adjacency_matrix(CoNLLCorpus corpus, int sentence_idx, KernelPerceptron kp);
void set_adjacency_matrix_fast(CoNLLCorpus corpus, int sentence_idx, KernelPerceptron kp, bool use_avg_alpha) ;

void free_FeaturedSentence(CoNLLCorpus corpus, int sentence_idx);

vector embedding_feature(FeaturedSentence sent, int from, int to, vector target);

#endif
