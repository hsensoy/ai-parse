//
//  vector.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "vector.h"
#include "memman.h"
#include "corpus.h"
#include "stringalgo.h"
#include <immintrin.h>


#include <float.h>
#include <math.h>

#define MIN(a,b) ((a) < (b) ? a : b)

/*
 * Next step is to use icc
 * source /opt/intel/composer_xe_2013_sp1/mkl/bin/mklvars.sh intel64
 * 
 * icc -mkl ...
 */


vector parse_vector(char *buff) {
    DArray *dim = split(buff, " ");

    vector v = vector_create(DArray_count(dim));

    for (int i = 0; i < DArray_count(dim); i++) {
        (v->data)[i] = atof((char*) DArray_get(dim, i));
        free((char*) DArray_get(dim, i));
    }

    DArray_destroy(dim);

    return v;

}

vector vector_create(size_t n) {
    vector v = (vector) malloc(sizeof (struct vector));
    check_mem(v);

    v->true_n = n;
    v->n = aligned_size(v->true_n);

    v->data = alloc_aligned(v->n);

    return v;

error:
    exit(1);
}

void vector_free(vector v) {
    if (v != NULL) {
        //mkl_free(v->data);
        free(v->data);
        free(v);
    }

}

void check_vector_len(const vector v1, const vector v2) {
    check(v1 != NULL, "v1 can not be NULL");
    check(v2 != NULL, "v2 can not be NULL");
    check(v1->n == v2->n && v1->true_n == v2->true_n, "v1(%ld:%ld) differs from v2(%ld:%ld) in size", v1->true_n, v1->n, v2->true_n, v2->n);

    return;
error:
    exit(1);
}

vector vconcat(vector target, const vector v) {
    check(v != NULL, "vector v can not be NULL");

    if (target == NULL) {

        target = vector_create(v->true_n);

        memcpy(target->data, v->data, sizeof (float) * (v->n));
    } else {
        vector temp = target;

        target = vector_create(v->true_n + target->true_n);

        memcpy(target->data, temp->data, sizeof (float) * temp->true_n);
        memcpy(target->data + temp->true_n, v->data, sizeof (float) * v->true_n);

        vector_free(temp);
    }

    return target;

error:
    exit(EXIT_FAILURE);
}

void vprint(vector v) {

    log_info("%ld:\t", v->true_n);

    for (int _v = 0; _v < v->true_n; _v++)
        log_info("%f ", v->data[_v]);
}

void vadd(vector target, const vector src, float mult) {
    check_vector_len(target, src);

    #pragma ivdep
    for (int i = 0; i < target->true_n; i++) {
        (target->data)[i] += mult * (src->data)[i];
    }
}

float vdot(vector v1, vector v2) {
    check_vector_len(v1, v2);

    
    float sum = 0.0;

    for (int i = 0; i < v1->true_n; i++)
        sum += (v1->data)[i] * (v2->data)[i];

    return sum;
    
    
    //return cblas_sdot(v1->true_n, v1->data,1, v2->data,1);
}

void vdiv(vector v, int div) {

    for (int i = 0; i < v->n; i++) {
        (v->data)[i] /= div;
    }
}

void vnorm(vector v, size_t n) {

    float len = 0.;

    for (int i = 0; i < v->n; i++) {
        len += (v->data)[i] * (v->data)[i];
    }

    len = sqrt(len);

    if (len >= FLT_EPSILON) {
        for (int i = 0; i < v->n; i++) {
            (v->data)[i] /= len;
        }
    }
}

vector vquadratic(vector target, vector src, float d) {
    float sqrt_of_2 = sqrt(2.);
    float sqrt_of_2d = sqrt(2. * d);

    int vquad_indx = 0;
    
    if (target == NULL) {
        vector vquad = vector_create((src->true_n * (src->true_n + 3)) / 2);

        for (int i = 0; i < src->true_n; i++) {
            vquad->data[vquad_indx++] = src->data[i] * src->data[i];


            for (int j = i + 1; j < src->true_n; j++) {


                vquad->data[vquad_indx++] = sqrt_of_2 * src->data[i] * src->data[j];


            }

            vquad->data[vquad_indx++] = sqrt_of_2d * src->data[i];
        }

        vector_free(src);

        return vquad;
    } else {

        
        for (int i = 0; i < src->true_n; i++) {
            target->data[vquad_indx++] = src->data[i] * src->data[i];


            for (int j = i + 1; j < src->true_n; j++) {


                target->data[vquad_indx++] = sqrt_of_2 * src->data[i] * src->data[j];


            }

            target->data[vquad_indx++] = sqrt_of_2d * src->data[i];
        }
         
        vector_free(src);

        return target;
    }
}

/*
float* continous_vector( int from, int to, FeaturedSentence sentence ){
    float *vect = NULL;
    float *scode=NULL;
    
    if (from != to && to != 0){
        size_t slen = sentence->scode_length;
        
        vect = alloc_aligned(aligned_size(slen * NUM_SCODE_FEATURES));    // p-1 p p+1 c-1 c c+1 and combinations of them
        float *pvect_cvect = alloc_aligned(aligned_size(slen));
        float *p_pP1_cM1_c = alloc_aligned(aligned_size(slen));
        float *p_pP1_c_cP1 = alloc_aligned(aligned_size(slen));
        float *pM1_p_cM1_c = alloc_aligned(aligned_size(slen));
        float *pM1_p_c_cP1 = alloc_aligned(aligned_size(slen));
        
        float *pvect_cvectN = alloc_aligned(aligned_size(slen));
        float *p_pP1_cM1_cN = alloc_aligned(aligned_size(slen));
        float *p_pP1_c_cP1N = alloc_aligned(aligned_size(slen));
        float *pM1_p_cM1_cN = alloc_aligned(aligned_size(slen));
        float *pM1_p_c_cP1N = alloc_aligned(aligned_size(slen));
        
        if (from != 0){
            scode = (float*)DArray_get(sentence->scode, from-1);
            
            if (scode != NULL){
                memcpy(vect, scode, sizeof(float) * slen);
                
                vadd(pM1_p_cM1_c, scode,1.,slen);
                vadd(pM1_p_c_cP1, scode,1.,slen);
                
                vadd(pM1_p_cM1_cN, scode,1.,slen);
                vadd(pM1_p_c_cP1N, scode,1.,slen);
            }
        }
        
        scode = (float*)DArray_get(sentence->scode, from);
        
        if (scode != NULL){
            memcpy(vect+slen, scode, sizeof(float) * slen);
            
            vadd(pvect_cvect, scode, 1., slen);
            vadd(p_pP1_cM1_c, scode,1.,slen);
            vadd(p_pP1_c_cP1, scode,1.,slen);
            vadd(pM1_p_cM1_c, scode,1.,slen);
            vadd(pM1_p_c_cP1, scode,1.,slen);
            
            vadd(pvect_cvectN, scode, 1., slen);
            vadd(p_pP1_cM1_cN, scode,1.,slen);
            vadd(p_pP1_c_cP1N, scode,1.,slen);
            vadd(pM1_p_cM1_cN, scode,1.,slen);
            vadd(pM1_p_c_cP1N, scode,1.,slen);
        }
        
        if (from != sentence->length - 1 && from != 0){
            scode = (float*)DArray_get(sentence->scode, from+1);
            
            if (scode != NULL){
                memcpy(vect+ 2 * slen, scode, sizeof(float) * slen);
                
                vadd(p_pP1_cM1_c, scode,1.,slen);
                vadd(p_pP1_c_cP1, scode,1.,slen);
                
                vadd(p_pP1_cM1_cN, scode,1.,slen);
                vadd(p_pP1_c_cP1N, scode,1.,slen);
            }
        }
        
        if (to != 0){
            scode = (float*)DArray_get(sentence->scode, to-1);
            
            if (scode != NULL){
                memcpy(vect+3 * slen, scode, sizeof(float) * slen);
                
                vadd(p_pP1_cM1_c, scode,1.,slen);
                vadd(pM1_p_cM1_c, scode,1.,slen);
                
                vadd(p_pP1_cM1_cN, scode,-1.,slen);
                vadd(pM1_p_cM1_cN, scode,-1.,slen);
            }
        }
        
        scode = (float*)DArray_get(sentence->scode, to);
        
        if (scode != NULL){
            memcpy(vect + 4 * slen, scode , sizeof(float) * slen);
            vadd(pvect_cvect, scode, 1., slen);
            vadd(p_pP1_cM1_c, scode,1.,slen);
            vadd(p_pP1_c_cP1, scode,1.,slen);
            vadd(pM1_p_cM1_c, scode,1.,slen);
            vadd(pM1_p_c_cP1, scode,1.,slen);
            
            vadd(pvect_cvectN, scode, -1., slen);
            vadd(p_pP1_cM1_cN, scode,-1.,slen);
            vadd(p_pP1_c_cP1N, scode,-1.,slen);
            vadd(pM1_p_cM1_cN, scode,-1.,slen);
            vadd(pM1_p_c_cP1N, scode,-1.,slen);
        }
        
        if (to != sentence->length - 1){
            scode = (float*)DArray_get(sentence->scode, to+1);
            
            if (scode != NULL){
                memcpy(vect+5 * slen, scode, sizeof(float) * slen);
                
                vadd(p_pP1_c_cP1, scode,1.,slen);
                vadd(pM1_p_c_cP1, scode,1.,slen);
                
                vadd(p_pP1_c_cP1N, scode,-1.,slen);
                vadd(pM1_p_c_cP1N, scode,-1.,slen);
            }
        }
        
        
        if (from != 0){
            int left_context, right_context;
            if (from < to){
                left_context = from;
                right_context = to;
            }else{
                left_context = to;
                right_context = from;
            }
            
            float *temp = alloc_aligned(slen);
            
            int n = right_context - 1 - left_context - 1 + 1;
            for(int j = left_context + 1; j < right_context;j++){
                scode = (float*)DArray_get(sentence->scode, j);
                
                if (scode!= NULL)
                    vadd(temp, scode, 1./(MIN(j-left_context, right_context-j)), slen);
            }
            
            if (n > 0)
                memcpy(vect+6*slen, temp, sizeof(float) * slen);
            
            free(temp);
        }
        
        
         vdiv(pvect_cvect, 2, slen);
         //  vnorm(pvect_cvect, slen);
         memcpy(vect+7*slen, pvect_cvect, sizeof(float) * slen);
         
         vdiv(p_pP1_cM1_c, 4, slen);
         //  vnorm(p_pP1_cM1_c, slen);
         memcpy(vect+8*slen, p_pP1_cM1_c, sizeof(float) * slen);
         
         vdiv(p_pP1_c_cP1, 4, slen);
         //    vnorm(p_pP1_c_cP1, slen);
         memcpy(vect+9*slen, p_pP1_c_cP1, sizeof(float) * slen);
         
         vdiv(pM1_p_cM1_c, 4, slen);
         //      vnorm(pM1_p_cM1_c, slen);
         memcpy(vect+10*slen, pM1_p_cM1_c, sizeof(float) * slen);
         
         vdiv(pM1_p_c_cP1, 4, slen);
         //        vnorm(pM1_p_c_cP1, slen);
         memcpy(vect+11*slen, pM1_p_c_cP1, sizeof(float) * slen);
         
         //
         
         vdiv(pvect_cvectN, 2, slen);
         //  vnorm(pvect_cvect, slen);
         memcpy(vect+12*slen, pvect_cvectN, sizeof(float) * slen);
         
         vdiv(p_pP1_cM1_cN, 4, slen);
         //  vnorm(p_pP1_cM1_c, slen);
         memcpy(vect+13*slen, p_pP1_cM1_cN, sizeof(float) * slen);
         
         vdiv(p_pP1_c_cP1N, 4, slen);
         //    vnorm(p_pP1_c_cP1, slen);
         memcpy(vect+14*slen, p_pP1_c_cP1N, sizeof(float) * slen);
         
         vdiv(pM1_p_cM1_cN, 4, slen);
         //      vnorm(pM1_p_cM1_c, slen);
         memcpy(vect+15*slen, pM1_p_cM1_cN, sizeof(float) * slen);
         
         vdiv(pM1_p_c_cP1N, 4, slen);
         //        vnorm(pM1_p_c_cP1, slen);
         memcpy(vect+16*slen, pM1_p_c_cP1N, sizeof(float) * slen);
         
        
        
        free(pvect_cvect);
        free(p_pP1_cM1_c);
        free(p_pP1_c_cP1);
        free(pM1_p_cM1_c);
        free(pM1_p_c_cP1);
        
        free(pvect_cvectN);
        free(p_pP1_cM1_cN);
        free(p_pP1_c_cP1N);
        free(pM1_p_cM1_cN);
        free(pM1_p_c_cP1N);
    }
    
    
    return vect;
    
error:
    exit(1);
}

float* continous_vector3( int from, int to, FeaturedSentence sentence ){
    float *vect = NULL;
    float *scode=NULL;
    
    if (from != to && to != 0){
        vect = alloc_aligned(aligned_size(25*7));    // p-1 p p+1 c-1 c c+1 sum(betweens)
        
        if (from != 0 && from-1 != to)
            memcpy(vect, (float*)DArray_get(sentence->scode, from-1), sizeof(float) * 25);
        
        
        scode = (float*)DArray_get(sentence->scode, from);
        
        memcpy(vect+25, scode, sizeof(float) * 25);
        
        if (from != sentence->length - 1 && from +1 != to)
            memcpy(vect+50, (float*)DArray_get(sentence->scode, from+1), sizeof(float) * 25);
        
        if (to != 0 && from != to -1)
            memcpy(vect+75, (float*)DArray_get(sentence->scode, to-1), sizeof(float) * 25);
        
        memcpy(vect+100, (float*)DArray_get(sentence->scode, to), sizeof(float) * 25);
        
        if (to != sentence->length - 1 && from != to +1)
            memcpy(vect+125, (float*)DArray_get(sentence->scode, to+1), sizeof(float) * 25);
        
        
        int left_context, right_context;
        if (from < to){
            left_context = from;
            right_context = to;
        }else{
            left_context = to;
            right_context = from;
        }
        
        float *temp = alloc_aligned(25);
        for(int j = left_context + 1; j < right_context;j++){
            vadd(temp, (float*)DArray_get(sentence->scode, j), 1., 25);
        }
        
        memcpy(vect+150, temp, sizeof(float) * 25);
        
        free(temp);
        
    }
    
    
    return vect;
    
error:
    exit(1);
}


 Between vector summation is normalized

float* continous_vector4( int from, int to, FeaturedSentence sentence ){
    float *vect = NULL;
    float *scode=NULL;
    
    if (from != to && to != 0){
        vect = alloc_aligned(aligned_size(25*7));    // p-1 p p+1 c-1 c c+1 sum(betweens)
        
        if (from != 0 && from-1 != to)
            memcpy(vect, (float*)DArray_get(sentence->scode, from-1), sizeof(float) * 25);
        
        
        scode = (float*)DArray_get(sentence->scode, from);
        
        memcpy(vect+25, scode, sizeof(float) * 25);
        
        if (from != sentence->length - 1 && from +1 != to)
            memcpy(vect+50, (float*)DArray_get(sentence->scode, from+1), sizeof(float) * 25);
        
        if (to != 0 && from != to -1)
            memcpy(vect+75, (float*)DArray_get(sentence->scode, to-1), sizeof(float) * 25);
        
        memcpy(vect+100, (float*)DArray_get(sentence->scode, to), sizeof(float) * 25);
        
        if (to != sentence->length - 1 && from != to +1)
            memcpy(vect+125, (float*)DArray_get(sentence->scode, to+1), sizeof(float) * 25);
        
        
        int left_context, right_context;
        if (from < to){
            left_context = from;
            right_context = to;
        }else{
            left_context = to;
            right_context = from;
        }
        
        float *temp = alloc_aligned(25);
        
        int n = right_context - 1 - left_context - 1 + 1;
        for(int j = left_context + 1; j < right_context;j++){
            vadd(temp, (float*)DArray_get(sentence->scode, j), 1., 25);
        }
        
        if (n > 0)
            vdiv(temp, n, 25);
        memcpy(vect+150, temp, sizeof(float) * 25);
        
        free(temp);
        
    }
    
    
    return vect;
    
error:
    exit(1);
}
 */
