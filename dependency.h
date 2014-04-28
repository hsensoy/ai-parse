//
//  dependency.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 13/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_dependency_h
#define Perceptron_GLM_NLP_Tasks_dependency_h

#include "darray.h"
#include "hashmap.h"
#include "corpus.h"
#include "datastructure.h"
#include <stdbool.h>

enum FeatureGroup {
    pword_ppos = 0,
    pword = 1,
    ppos = 2,
    cword_cpos = 3,
    cword = 4,
    cpos = 5,
    pword_ppos_cword_cpos = 6,
    ppos_cword_cpos = 7,
    pword_ppos_cpos = 8,
    pword_ppos_cword = 9,
    pword_cword = 10,
    ppos_cpos = 11,
    ppos_pposP1_cposM1_cpos = 12,
    pposM1_ppos_cposM1_cpos = 13,
    ppos_pposP1_cpos_cposP1 = 14,
    pposM1_ppos_cpos_cposP1 = 15,
    ppos_bpos_cpos = 16
};

typedef enum FeatureGroup FeatureGroup;


IntegerIndexedFeatures IntegerIndexedFeatures_create();

struct FeatureKey {
    FeatureGroup grp;
    char* value;
};

typedef struct FeatureKey* FeatureKey;

/**
 n1: n1++ if feature defines an arc
 n2: n2++ if feature occurs for a potential arc
 */
struct FeatureValue {
    uint32_t feature_id;
    uint32_t n1;
    uint32_t n2;
};

typedef struct FeatureValue* FeatureValue;

struct PerceptronModel {
    IntegerIndexedFeatures features;

    vector discrete_w;
    vector discrete_w_avg;
    vector discrete_w_temp;

    vector embedding_w;
    vector embedding_w_avg;
    vector embedding_w_temp;
    vector embedding_w_best;

    int best_numit;

    int c;

    size_t n;

    bool use_discrete_features;
};

typedef struct PerceptronModel* PerceptronModel;



/**
    FeatureKey, FeatureValue creation functions.
 */
FeatureKey FeatureKey_create(FeatureGroup group, char* value);
FeatureValue FeatureValue_create(uint32_t fid);


int feature_equal(void *k1, void *k2);
uint32_t feature_hash(void *f);

PerceptronModel PerceptronModel_create(CoNLLCorpus training, IntegerIndexedFeatures iif);
void PerceptronModel_free(PerceptronModel model);

void train_perceptron_parser(PerceptronModel mdl, const CoNLLCorpus corpus, int numit, int max_rec);
void train_perceptron_once(PerceptronModel mdl, const CoNLLCorpus corpus, int max_rec);
double test_perceptron_parser(PerceptronModel mdl, const CoNLLCorpus corpus, bool exclude_punct, bool use_temp);

void fill_features(Hashmap *featuremap, DArray *farr, int from, int to, FeaturedSentence sentence);
void parse_and_dump(PerceptronModel mdl, FILE *fp, CoNLLCorpus corpus);

/**
 * 
 * @param fp File pointer of model file
 * @param edimension Word embedding dimensions
 * @param w Weight vector to be stored
 * @param best_numit Number of iterations for best performance.
 */
void dump_PerceptronModel(FILE *fp, int edimension, vector w, int best_numit);

int* parse(FeaturedSentence sent);
int* get_parents(const FeaturedSentence sent);
int nmatch(const int* model, const int* empirical, int length);
void printfarch(int *parent, int len);

void mark_best_PerceptronModel(PerceptronModel model, int numit);

#endif
