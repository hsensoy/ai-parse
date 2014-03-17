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
#include <stdbool.h>

#define EXAMPLE_CONLL_DIR "/Users/husnusensoy/uparse/data/nlp/treebank/treebank-2.0/combined/conll"

#define  STOP  "<STOP>"
#define START  "*"
//static const char* ROOT = "root";

#define MAX_SENT_LENGTH 20

struct Word {
    int id;
    int parent;
    char *postag;
    char *form;
    char *relation;

    vector embedding;
};

typedef struct Word* Word;

//Word parse_word( char* line, bool read_vector );

struct FeatureVector {
    DArray *discrete_v;
    vector continous_v;
};

typedef struct FeatureVector* FeatureVector;

struct FeatureMatrix{
    FeatureVector** matrix_data;
    uint16_t size;
    uint32_t embedding_length;
    bool has_discrete_features;
};

typedef  struct FeatureMatrix* FeatureMatrix;

FeatureVector FeatureVector_create(bool has_discrete_features, uint32_t embedding_length);

struct FeaturedSentence {
    uint8_t section;

    DArray* words;
    int length;
    //DArray* postags;
    //DArray* embedding;
    //DArray* parents;

    //DArray ***feature_matrix;   // For each potential link from-->to you have a set of features.
    //FeatureVector **feature_matrix;
    
    /**
     * This is simply a reference to actual one used in CoNLLCorpus structure.
     * This allows us to remote 
     */
    FeatureMatrix feature_matrix_ref;           
    

    float **adjacency_matrix; // Score of each potential link betweek words
};


typedef struct FeaturedSentence* FeaturedSentence;

enum EmbeddingTranformation{
    QUADRATIC,
    LINEAR
};

struct CoNLLCorpus {
    const char *base_dir;
    DArray* sections;

    DArray *sentences;

    DArray *embedding_pattern_parts;
    bool hasembeddings;
    DArray *disrete_patterns_parts;

    Word Root;
    size_t embedding_length;
    size_t transformed_embedding_length;
    
    /**
     * Create a feature matrix of length 300 to guarantee all sentence length with high probability.
     */
    FeatureMatrix feature_matrix_singleton;
    enum EmbeddingTranformation embedding_transform;
};

typedef struct CoNLLCorpus* CoNLLCorpus;



CoNLLCorpus create_CoNLLCorpus(const char* base_dir, DArray *sections, const char *embedding_pattern, enum EmbeddingTranformation transform, DArray* discrete_patterns);
void read_corpus(CoNLLCorpus coprus, bool build_feature_matrix);
void free_CoNLLCorpus(CoNLLCorpus corpus);



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
void free_feature_matrix(CoNLLCorpus corpus, int sentence_idx);

void build_adjacency_matrix(CoNLLCorpus corpus, int sentence_idx, vector embeddings_w, vector discrete_w);

#endif