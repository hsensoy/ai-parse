//
//  dependency.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 13/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//
#include "dependency.h"
#include "stringalgo.h"
#include "memman.h"
#include "vector.h"

#include <math.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>

#define GAMMA 0.5
/*
float rbf(const float *v1, const float *v2, size_t  n){
    float result = 0.0;
    
    for (int i = 0; i < n; i++) {
        float delta = v1[i] - v2[i];
        result += delta * delta;
    }
    
    return exp(-GAMMA * result);
}*/

#define KERNEL vdot
#define GET_CONTINOUS_VECTOR big_vector

float* big_vector(int from, int to, FeaturedSentence sentence) {
    return NULL;
}

/*
float* big_vector( int from, int to, FeaturedSentence sentence ){
    float *vect = NULL;
    float *scode=NULL;
    
    if (from != to && to != 0){
        size_t slen = sentence->scode_length;
        
        vect = alloc_aligned(aligned_size(slen * NUM_SCODE_FEATURES));    // p-1 p p+1 c-1 c c+1 and combinations of them

        if (from != 0){
            scode = (float*)DArray_get(sentence->scode, from-1);
            
            if (scode != NULL)
                memcpy(vect, scode, sizeof(float) * slen);
        }
        
        scode = (float*)DArray_get(sentence->scode, from);
        
        if (scode != NULL)
            memcpy(vect + slen, scode, sizeof(float) * slen);
            
        
        if (from != sentence->length - 1 && from != 0){
            scode = (float*)DArray_get(sentence->scode, from+1);
            
            if (scode != NULL)
                memcpy(vect + 2 * slen, scode, sizeof(float) * slen);
        }
        
        if (to != 0){
            scode = (float*)DArray_get(sentence->scode, to-1);
            
            if (scode != NULL)
                memcpy(vect + 3 * slen, scode, sizeof(float) * slen);
        }
        
        scode = (float*)DArray_get(sentence->scode, to);
        
        if (scode != NULL)
            memcpy(vect + 4 * slen, scode , sizeof(float) * slen);
        
        if (to != sentence->length - 1){
            scode = (float*)DArray_get(sentence->scode, to+1);
            
            if (scode != NULL)
                memcpy(vect + 5 * slen, scode, sizeof(float) * slen);
        }
    }
    
    
    return vect;
    
error:
    exit(1);
}
 */




FeatureValue FeatureValue_create(uint32_t fid) {

    FeatureValue fvalue = (FeatureValue) malloc(sizeof (struct FeatureValue));
    check_mem(fvalue);

    fvalue->feature_id = fid;
    fvalue->n1 = 0;
    fvalue->n2 = 0;

    return fvalue;
error:
    log_err("FeatureValue allocation error");
    exit(1);
}

FeatureKey FeatureKey_create(FeatureGroup group, char* value) {

    FeatureKey f = (FeatureKey) malloc(sizeof (struct FeatureKey));
    check_mem(f);

    f->grp = group;
    f->value = strdup(value);

    return f;

error:
    log_err("Error in creating feature key %d:%s", group, value);
    exit(1);
}

/**
 Returns 0 for equality and 1 for inequality
 */
int feature_equal(void *k1, void *k2) {
    FeatureKey f1 = (FeatureKey) k1;
    FeatureKey f2 = (FeatureKey) k2;

    return !(f1->grp == f2->grp && (strcmp(f1->value, f2->value) == 0));
}

uint32_t feature_hash(void *f) {
    uint32_t prime = 31;
    uint32_t result = 1;

    FeatureKey feat = (FeatureKey) f;

    result = prime * result + feat->grp;
    result = prime * result + default_hash(feat->value);

    return result;
}

IntegerIndexedFeatures IntegerIndexedFeatures_create() {
    IntegerIndexedFeatures imap = (IntegerIndexedFeatures) malloc(sizeof (struct IntegerIndexedFeatures));
    check_mem(imap);

    imap->feature_id = 0;
    imap->map = Hashmap_create(feature_equal, feature_hash);


    return imap;
error:
    log_err("IntegerIndexedFeatures allocation error");
    exit(1);
}

void free_sentence_structures(FeaturedSentence sentence) {
    for (int i = 0; i < DArray_count(sentence->words); i++) {
        free(sentence->adjacency_matrix[i]);
    }

    free(sentence->adjacency_matrix);

    sentence->adjacency_matrix = NULL;
}

void destroy_parent_table(int***** table, int length) {
    for (int i = 0; i < length; i++) {

        for (int j = 0; j < length; j++) {


            for (int k = 0; k < 2; k++) {


                for (int t = 0; t < 2; t++) {
                    free(table[i][j][k][t]);
                }

                free(table[i][j][k]);
            }

            free(table[i][j]);
        }

        free(table[i]);
    }

    free(table);
}

void destroy_score_table(double**** table, int length) {

    for (int i = 0; i < length; i++) {

        for (int j = 0; j < length; j++) {

            for (int k = 0; k < 2; k++) {
                free(table[i][j][k]);
            }

            free(table[i][j]);
        }

        free(table[i]);
    }


    free(table);
}

double**** init_score_table(int length) {

    double**** table = (double****) malloc(sizeof (double***) * length);
    check_mem(table);

    for (int i = 0; i < length; i++) {
        table[i] = (double***) malloc(sizeof (double**) * length);
        check_mem(table[i]);

        for (int j = 0; j < length; j++) {
            table[i][j] = (double**) malloc(sizeof (double*) * 2);
            check_mem(table[i][j]);

            for (int k = 0; k < 2; k++) {
                table[i][j][k] = (double*) calloc(2, sizeof (double));

                check_mem(table[i][j][k]);
            }
        }
    }

    return table;

error:
    log_err("Error in score table allocation");
    exit(1);
}

int***** init_parent_table(int length) {

    int***** table = (int*****) malloc(sizeof (int****) * length);
    check_mem(table);

    for (int i = 0; i < length; i++) {
        table[i] = (int****) malloc(sizeof (int***) * length);
        check_mem(table[i]);

        for (int j = 0; j < length; j++) {
            table[i][j] = (int***) malloc(sizeof (int**) * 2);
            check_mem(table[i][j]);

            for (int k = 0; k < 2; k++) {
                table[i][j][k] = (int**) malloc(sizeof (int*) * 2);

                check_mem(table[i][j][k]);

                for (int t = 0; t < 2; t++) {
                    table[i][j][k][t] = (int*) calloc(length, sizeof (int));

                    for (int q = 0; q < length; q++)
                        table[i][j][k][t][q] = -1;

                    check_mem(table[i][j][k][t]);
                }
            }
        }
    }

    return table;

error:
    log_err("Error in score table allocation");
    exit(1);
}

void addAll(int *t, int *s, int l) {

    for (int i = 1; i < l; i++) {


        check((t[i] == -1 && s[i] != -1) || s[i] == -1, "Target can not have parent %d <- %d", i, t[i]);

        if (s[i] != -1)
            t[i] = s[i];
    }

    return;
error:
    exit(1);
}

void add(int *t, int parent, int child) {

    check(t[child] == -1, "Target can not have parent %d <- %u", child, t[child]);

    t[child] = parent;

    return;
error:
    exit(1);
}

int* parse(FeaturedSentence sent) {
    double ****E = init_score_table(DArray_count(sent->words) + 1);
    int *****P = init_parent_table(DArray_count(sent->words) + 1);

    for (int m = 1; m < DArray_count(sent->words) + 1; m++) {
        for (int s = 0; s < DArray_count(sent->words) + 1; s++) {
            int t = s + m;

            if (t > DArray_count(sent->words))
                break;
            else {
                double bestscore = NEGATIVE_INFINITY;
                int bestq = -1;
                for (int q = s; q < t; q++) {
                    double score = E[s][q][1][0] + E[q + 1][t][0][0]
                            + sent->adjacency_matrix[t][s];

                    if (score >= bestscore) {
                        bestscore = score;
                        bestq = q;
                    }
                }

                E[s][t][0][1] = bestscore;

                addAll(P[s][t][0][1], P[s][bestq][1][0], DArray_count(sent->words) + 1);
                addAll(P[s][t][0][1], P[bestq + 1][t][0][0], DArray_count(sent->words) + 1);


                add(P[s][t][0][1], t, s);

                bestscore = NEGATIVE_INFINITY;
                bestq = -1;
                for (int q = s; q < t; q++) {
                    double score = E[s][q][1][0] + E[q + 1][t][0][0]
                            + sent->adjacency_matrix[s][t];

                    if (score >= bestscore) {
                        bestscore = score;
                        bestq = q;
                    }
                }

                E[s][t][1][1] = bestscore;


                addAll(P[s][t][1][1], P[s][bestq][1][0], DArray_count(sent->words) + 1);
                addAll(P[s][t][1][1], P[bestq + 1][t][0][0], DArray_count(sent->words) + 1);


                add(P[s][t][1][1], s, t);

                bestscore = NEGATIVE_INFINITY;
                bestq = -1;
                for (int q = s; q < t; q++) {
                    double score = E[s][q][0][0] + E[q][t][0][1];

                    if (score >= bestscore) {
                        bestscore = score;
                        bestq = q;
                    }
                }

                E[s][t][0][0] = bestscore;


                addAll(P[s][t][0][0], P[s][bestq][0][0], DArray_count(sent->words) + 1);
                addAll(P[s][t][0][0], P[bestq][t][0][1], DArray_count(sent->words) + 1);


                bestscore = NEGATIVE_INFINITY;
                bestq = -1;
                for (int q = s + 1; q <= t; q++) {
                    double score = E[s][q][1][1] + E[q][t][1][0];

                    if (score >= bestscore) {
                        bestscore = score;
                        bestq = q;
                    }
                }

                E[s][t][1][0] = bestscore;


                addAll(P[s][t][1][0], P[s][bestq][1][1], DArray_count(sent->words) + 1);
                addAll(P[s][t][1][0], P[bestq][t][1][0], DArray_count(sent->words) + 1);
            }

        }
    }

    debug("Out of loop");


    //DArray* r = A[0][sent->length - 1][1][0];

    int* r = P[0][DArray_count(sent->words)][1][0];

    //check(DArray_count(r) == sent->length-1,
    //  "Number of arcs in dependency parse tree(%d) should be equal to sentence length(%d)", DArray_count(r), sent->length-1);

    int *parents = (int*) malloc((DArray_count(sent->words) + 1) * sizeof (int));
    check_mem(parents);


    for (int i = 0; i < DArray_count(sent->words) + 1; i++)
        parents[i] = r[i];

    //exit(1);

    destroy_score_table(E, DArray_count(sent->words) + 1);
    //destroy_arc_table(A, sent->length);

    destroy_parent_table(P, DArray_count(sent->words) + 1);

    //    DArray_clear_destroy(gc);


    return parents;

error:
    log_err("Error in allocating gc array");
    exit(1);
}

int* get_parents(const FeaturedSentence sent) {

    int *parents = (int*) malloc(sizeof (int) * (DArray_count(sent->words) + 1));
    check_mem(parents);

    parents[0] = -1;
    for (int i = 0; i < DArray_count(sent->words); i++) {
        Word w = (Word) DArray_get(sent->words, i);
        parents[i + 1] = w->parent;
    }

    return parents;

error:
    log_err("Error in allocating parents");
    exit(1);
}

int nmatch(const int* model, const int* empirical, int length) {

    int nmatch = 0;

    for (int i = 1; i <= length; i++) {
        if (model[i] == empirical[i]) {
            nmatch++;

            debug("%d -> %d is correct\n", model[i], i);
        }
    }

    return nmatch;
}

void add_feature(FeatureGroup group, char* value, Hashmap *featuremap, DArray *list) {
    struct FeatureKey ro;
    ro.grp = group;
    ro.value = value;

    FeatureValue fv;
    fv = (FeatureValue) Hashmap_get(featuremap, &ro);

    if (fv != NULL)
        DArray_push(list, &(fv->feature_id));
}

/**
 Fill the features DArray for [from][to cell
 */
void fill_features(Hashmap *featuremap, DArray *farr, int from, int to, FeaturedSentence sentence) {
    char buffer[255];

    if (from != to && to != 0) {

        Word pword_w = (Word) DArray_get(sentence->words, from);
        Word cword_w = (Word) DArray_get(sentence->words, to);


        join(buffer, 2, pword_w->form, pword_w->postag);
        add_feature(pword_ppos, buffer, featuremap, farr);

        add_feature(pword, pword_w->form, featuremap, farr);
        add_feature(ppos, pword_w->postag, featuremap, farr);


        join(buffer, 2, cword_w->form, cword_w->postag);
        add_feature(cword_cpos, buffer, featuremap, farr);

        add_feature(cword, cword_w->form, featuremap, farr);
        add_feature(cpos, cword_w->postag, featuremap, farr);


        join(buffer, 4, pword_w->form, pword_w->postag, cword_w->form, cword_w->postag);
        add_feature(pword_ppos_cword_cpos, buffer, featuremap, farr);


        join(buffer, 3, pword_w->postag, cword_w->form, cword_w->postag);
        add_feature(ppos_cword_cpos, buffer, featuremap, farr);

        join(buffer, 3, pword_w->form, pword_w->postag, cword_w->postag);
        add_feature(pword_ppos_cpos, buffer, featuremap, farr);


        join(buffer, 3, pword_w->form, pword_w->postag, cword_w->form);
        add_feature(pword_ppos_cword, buffer, featuremap, farr);


        join(buffer, 2, pword_w->form, cword_w->form);
        add_feature(pword_cword, buffer, featuremap, farr);


        join(buffer, 2, pword_w->postag, cword_w->postag);
        add_feature(ppos_cpos, buffer, featuremap, farr);

        /*
        
char* pposP1_v;
char* pposM1_v;
char* cposM1_v;
char* cposP1_v;
if (from == DArray_count(sentence->words) - 1 )
    pposP1_v = STOP;
else
			
    pposP1_v = (char*)DArray_get(sentence->postags, from+1);
        
if (from == 0 )
    pposM1_v = START;
else
    pposM1_v = (char*)DArray_get(sentence->postags, from-1);
        
if (to == 0 )
    cposM1_v = START;
else
    cposM1_v = (char*)DArray_get(sentence->postags, to-1);
        
if (to == sentence->length - 1 )
    cposP1_v = STOP;
else
    cposP1_v = (char*)DArray_get(sentence->postags, to+1);
        
join(buffer, 4, ppos_v, pposP1_v, cposM1_v,cpos_v);
add_feature(ppos_pposP1_cposM1_cpos, buffer, featuremap,farr);
        
        
join(buffer, 4, pposM1_v, ppos_v, cposM1_v,cpos_v);
add_feature(pposM1_ppos_cposM1_cpos, buffer, featuremap,farr);
        
        
join(buffer, 4, ppos_v, pposP1_v, cpos_v,cposP1_v);
add_feature(ppos_pposP1_cpos_cposP1, buffer, featuremap,farr);
        
        
join(buffer, 4, pposM1_v, ppos_v, cpos_v,cposP1_v);
add_feature(pposM1_ppos_cpos_cposP1, buffer, featuremap,farr);
        
        
int left_context, right_context;
if (from < to){
    left_context = from;
    right_context = to;
}else{
    left_context = to;
    right_context = from;
}
        
for(int j = left_context + 1; j < right_context;j++){
    join(buffer, 3, DArray_get(sentence->postags, left_context), DArray_get(sentence->postags, j), DArray_get(sentence->postags, right_context));
            
    add_feature(ppos_bpos_cpos, buffer, featuremap, farr);
            
}
         * */
    }

}

PerceptronModel PerceptronModel_create(CoNLLCorpus training, IntegerIndexedFeatures iif) {

    PerceptronModel model = (PerceptronModel) malloc(sizeof (struct PerceptronModel));
    check_mem(model);

    model->c = 1;

    //FeatureMatrix mat = training->feature_matrix_singleton;

    if (training->hasembeddings) {
        model->embedding_w = vector_create(training->transformed_embedding_length);
        model->embedding_w_avg = vector_create(training->transformed_embedding_length);
        model->embedding_w_temp = vector_create(training->transformed_embedding_length);

        log_info("Embedding features of %ld dimension", model->embedding_w->true_n);
    } else {
        model->embedding_w = NULL;
        model->embedding_w_avg = NULL;
        model->embedding_w_temp = NULL;
    }

    if (training->disrete_patterns_parts) {
        model->discrete_w = vector_create(iif->feature_id);
        model->discrete_w_avg = vector_create(iif->feature_id);
        model->discrete_w_temp = vector_create(iif->feature_id);

        log_info("Discrete features of %ld dimension", model->discrete_w->true_n);
    } else {
        model->discrete_w = NULL;
        model->discrete_w_avg = NULL;
        model->discrete_w_temp = NULL;

    }

    return model;
error:
    exit(1);
}

void PerceptronModel_free(PerceptronModel model) {
    if (model->discrete_w) {
        vector_free(model->discrete_w);
        vector_free(model->discrete_w_avg);
        vector_free(model->discrete_w_temp);
    }

    if (model->embedding_w) {
        vector_free(model->embedding_w);
        vector_free(model->embedding_w_avg);
        vector_free(model->embedding_w_temp);
    }
}

void printfarch(int *parent, int len) {
    log_info("There are %d archs in total", len);
    for (int i = 1; i < len; i++) {
        log_info("%d->%d", parent[i], i);
    }
    log_info("\n");
}

void printfmatrix(float **mat, size_t n) {

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++)
            if (i == j || j == 0)
                printf("%s\t", "Inf");
            else
                printf("%4.2f\t", mat[i][j]);
        printf("\n");
    }
}

void printfembedding(FeatureVector **mat, size_t n) {

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++)
            if (i != j && j != 0) {
                log_info("%d -> %d", i, j);
                vprint(mat[i][j]->continous_v);
            }

        printf("\n");
    }
}


double dsecnd(){
    return 0.0;
}

void train_perceptron_once(PerceptronModel mdl, const CoNLLCorpus corpus) {
    long match = 0, total = 0;
    //size_t slen=0;

    double s_initial =  dsecnd();
    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        //log_info("Parsing sentence %d/%d", si+1, DArray_count(corpus));
        FeaturedSentence sent = (FeaturedSentence) DArray_get(corpus->sentences, si);

        debug("Building feature matrix for sentence %d", si);
        set_FeatureMatrix(NULL, corpus, si);

        //printfembedding(sent->feature_matrix, sent->length);

        debug("Building adjacency matrix for sentence %d", si);
        build_adjacency_matrix(corpus, si, mdl->embedding_w, NULL);

        //printfmatrix(sent->adjacency_matrix, sent->length);

        //log_info("Adjacency matrix construction is done");


        int *model = parse(sent);
        debug("Parsing sentence %d is done", si);
        int *empirical = get_parents(sent);

        /*
        log_info("Model:");
        printfarch(model, sent->length);
        log_info("Empirical:");
        printfarch(empirical, sent->length);
         */

        int nm = nmatch(model, empirical, sent->length);
        debug("Model matches %d arcs out of %d arcs", nm, sent->length);
        if (nm != sent->length) { // root has no valid parent.

            if (corpus->disrete_patterns_parts) {

                log_info("I have discrete features");

                DArray* model_features = DArray_create(sizeof (uint32_t), 16);
                DArray* empirical_features = DArray_create(sizeof (uint32_t), 16);

                for (int fi = 1; fi < sent->length; fi++) {
                    fill_features(mdl->features->map, model_features, model[fi], fi, sent);

                    fill_features(mdl->features->map, empirical_features, empirical[fi], fi, sent);
                }

                for (int i = 0; i < DArray_count(model_features); i++) {
                    uint32_t *fidx = (uint32_t *) DArray_get(model_features, i);

                    mdl->discrete_w->data[*fidx] -= 1.0;
                    mdl->discrete_w_avg->data[*fidx] -= (mdl->c) * 1.0;
                    mdl->discrete_w_temp->data[*fidx] -= 1.0;

                }

                for (int i = 0; i < DArray_count(empirical_features); i++) {
                    uint32_t *fidx = (uint32_t *) DArray_get(empirical_features, i);

                    mdl->discrete_w->data[*fidx] += 1.0;
                    mdl->discrete_w_avg->data[*fidx] += (mdl->c) * 1.0;

                    mdl->discrete_w_temp->data[*fidx] += 1.0;
                }

                DArray_destroy(model_features);
                DArray_destroy(empirical_features);
            }


            for (int i = 1; i <= sent->length; i++) {

                vector model_embedding = (sent->feature_matrix_ref->matrix_data[model[i]][i])->continous_v;
                vector real_embedding = (sent->feature_matrix_ref->matrix_data[empirical[i]][i])->continous_v;

                /*
                debug("Increasing vector");
                vprint(real_embedding);
				
                debug("Decreasing vector");
                vprint(model_embedding);
                 */

                check(model_embedding != NULL && empirical != NULL, "model_embedding/real_embedding can not be NULL at this stage ?!?");

                vadd(mdl->embedding_w, model_embedding, -1.0);
                vadd(mdl->embedding_w, real_embedding, 1.0);

                /*
                debug("Weight vector");
                vprint(mdl->embedding_w);
                 */
                vadd(mdl->embedding_w_temp, model_embedding, -1.0);
                vadd(mdl->embedding_w_temp, real_embedding, 1.0);

                vadd(mdl->embedding_w_avg, model_embedding, -(mdl->c));
                vadd(mdl->embedding_w_avg, real_embedding, (mdl->c));

                //free(real_embedding);
                //free(model_embedding);
            }
        }

        free_feature_matrix(corpus, si);

        mdl->c++;

        match += nm;
        total += (sent->length);

        if (si % 1000 == 1){
            log_info("Running training accuracy %lf (Elapsed %.5f sec)", (match * 1.) / total, ( dsecnd() - s_initial));
            
            s_initial =  dsecnd();
        }


        free(model);
        free(empirical);

        //free_sentence_structures(sent);
    }

    log_info("Running training accuracy %lf", (match * 1.) / total);

    if (corpus->disrete_patterns_parts) {
        for (int i = 0; i < mdl->n; i++) {
            //        mdl->w_avg[i] /= (numit * DArray_count(corpus));

            //mdl->w[i] -=(mdl->w_avg[i])/(mdl->c);

            mdl->discrete_w_temp->data[i] = mdl->discrete_w->data[i] - (mdl->discrete_w_avg->data[i]) / (mdl->c);
        }
    }

    //vadd(mdl->w_cont, mdl->w_cont_avg, -1./(mdl->c), SCODE_FEATURE_VECTOR_LENGTH);
    memcpy(mdl->embedding_w_temp->data, mdl->embedding_w->data, mdl->embedding_w->n * sizeof (float));
    vadd(mdl->embedding_w_temp, mdl->embedding_w_avg, -1. / (mdl->c));

    //    free(mdl->w);
    //    mdl->w = mdl->w_avg;

    //free_feature_matrix(sent);

    return;
error:
    exit(1);
}

void train_perceptron_parser(PerceptronModel mdl, const CoNLLCorpus corpus, int numit) {

    for (int n = 0; n < numit; n++) {

        log_info("%d iteration in parser training...", (n + 1));

        train_perceptron_once(mdl, corpus);
    }

    if (corpus->disrete_patterns_parts)
        memcpy(mdl->discrete_w->data, mdl->discrete_w_temp->data, mdl->n * sizeof (float));

    //FeaturedSentence f = (FeaturedSentence)DArray_get(corpus, 0);

    memcpy(mdl->embedding_w->data, mdl->embedding_w_temp->data, mdl->embedding_w->n * sizeof (float));
}

void parse_and_dump(PerceptronModel mdl, FILE *fp, CoNLLCorpus corpus) {
    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        FeaturedSentence sent = DArray_get(corpus->sentences, si);

        build_adjacency_matrix(corpus, si, mdl->embedding_w, NULL);

        int *model = parse(sent);


        for (int j = 1; j < sent->length; j++) {
            Word w = (Word) DArray_get(sent->words, j);
            int p = w->parent;

            char *form = w->form;
            char *postag = w->postag;


            fprintf(fp, "%d\t%s\t%s\t%d\t%d\n", j, form, postag, p, model[j]);
        }
        fprintf(fp, "\n");
    }
}

double test_perceptron_parser(PerceptronModel mdl, const CoNLLCorpus corpus, bool exclude_punct, bool use_temp) {

    int match = 0, total = 0;
    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        FeaturedSentence sent = DArray_get(corpus->sentences, si);

        debug("Generating feature matrix for sentence %d", si);
        set_FeatureMatrix(NULL, corpus, si);

        debug("Generating adj. matrix for sentence %d", si);
        if (use_temp) {
            debug("\tI will be using a weight vector of length %ld", mdl->embedding_w_temp->true_n);
            build_adjacency_matrix(corpus, si, mdl->embedding_w_temp, NULL);
        } else {
            debug("\tI will be using a weight vector of length %ld", mdl->embedding_w->true_n);
            build_adjacency_matrix(corpus, si, mdl->embedding_w, NULL);
        }

        debug("Now parsing sentence %d", si);
        int *model = parse(sent);


        debug("Now comparing actual arcs with model generated arcs for sentence %d (Last sentence is %d)", si, sent->length);
        for (int j = 0; j < sent->length; j++) {
            Word w = (Word) DArray_get(sent->words, j);
            int p = w->parent;
            char *postag = w->postag;

            debug("\tTrue parent of word %d (with %s:%s) is %d whereas estimated parent is %d", j, postag, w->form, p, model[j + 1]);

            if (exclude_punct) {
                if (strcmp(postag, ",") != 0 && strcmp(postag, ":") != 0 && strcmp(postag, ".") != 0 && strcmp(postag, "``") != 0 && strcmp(postag, "''") != 0) {

                    if (p == model[j + 1])
                        match++;

                    total++;
                }
            } else {
                if (p == model[j + 1])
                    match++;

                total++;
            }
        }


        debug("Releasing feature matrix for sentence %d", si);

        free_feature_matrix(corpus, si);


    }

    return (match * 1.) / total;
}
