//
//  vector.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_vector_h
#define Perceptron_GLM_NLP_Tasks_vector_h

#include <float.h>
#define NEGATIVE_INFINITY (-(FLT_MAX - 10.))

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
//#include "mkl.h"

#include "uthash.h"
#include "debug.h"


struct vector{
	size_t n;
	size_t true_n;
	float* data;
};

typedef struct vector* vector;

vector parse_vector( char *buff);

vector vector_create( size_t n );
void vector_free(vector);



/**
 * @brief Element-wise vector addition: target = target + mult * src
 * @param target Target vector on which src to be added.
 * @param src Vector to be added.
 * @param mult Multiplier of src before adding on target
 */
void vadd(vector target, const vector src, float mult);

float vdot(vector v1, vector v2);

void vdiv(vector v, int div);
void vnorm(vector v, size_t n);

void vprint(vector v);

vector vlinear(vector target, vector src);
vector vquadratic(vector target, vector src, float d);

/**
 * @brief Vector concatenation like np.concatenate
 * @param target 
 * @param v
 * @return vector of true_n = target->true_n + v->true_n aligned to 32 bytes
 */
vector vconcat(vector target, const vector v);


enum Kernel{
    KLINEAR,
    KPOLYNOMIAL 
};

typedef struct {
    uint32_t sentence_idx;
    uint16_t from;
    uint16_t to;
} alpha_key_t;


typedef struct alpha{
    UT_hash_handle hh;
        
    float alpha;
    vector v;
    
    uint32_t sentence_idx;
    uint16_t from;
    uint16_t to;
} alpha_t;

#define IS_ARC_VALID(from,to, length) check((from) != (to) && (from) <= (length) && (from) >= 0 && (to)>= 1 && (to) <= (length), "Arc between suspicious words %d to %d for sentence length %d", (from), (to), (length))

alpha_t* create_alpha_idx(int sentence_idx, int from, int to, int length);

/**
 * 
 * @param v1
 * @param v2
 * @return v1.v2
 */
static inline float linear(vector v1, vector v2){
    return vdot(v1,v2);
}

/**
 * 
 * @param v1
 * @param v2
 * @param d 
 * @param n
 * @return (v1.v2 + d )^n
 */ 
static inline float polynomial(vector v1, vector v2, float d, int n){
    return pow( vdot(v1,v2) + d ,n) ;
}
#endif
