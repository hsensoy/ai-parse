//
//  datastructure.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_datastructure_h
#define Perceptron_GLM_NLP_Tasks_datastructure_h
#include <stdbool.h>
#include "vector.h"
#include "hashmap.h"
#include "mkl.h"

struct IntegerIndexedFeatures{
    Hashmap *map;
    uint32_t feature_id;
};

typedef struct IntegerIndexedFeatures* IntegerIndexedFeatures;


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
    

    float **adjacency_matrix; // Score of each potential link between words
};


typedef struct FeaturedSentence* FeaturedSentence;


typedef struct {
    uint32_t sentence_idx;
    uint16_t from;
    uint16_t to;
} alpha_key_t;


typedef struct alpha{
    UT_hash_handle hh;
        
    int idx;
    
    uint32_t sentence_idx;
    uint16_t from;
    uint16_t to;
} alpha_t;

enum Kernel{
    KLINEAR,
    KPOLYNOMIAL 
};


struct KernelPerceptron{
    float bias;
    int power;
    
    size_t M;
    float* alpha;
    float* alpha_avg;
    float* kernel_matrix;
    
    enum Kernel kernel;
    
    size_t N;
    
    
    float* beta;
    

    int c;
    
    
    float* best_alpha_avg;
    int best_numit;
    float* best_kernel_matrix;
    size_t best_m;
 
    alpha_t *arch_to_index_map;
};

typedef struct KernelPerceptron* KernelPerceptron;



#endif
