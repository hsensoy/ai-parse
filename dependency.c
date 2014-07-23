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
#define PARSER_RATE_VERBOSITY 250       // Log parser rate in every PARSER_RATE_VERBOSITY sentences.

/**
 * ai-parse.c file for actual storage allocation for those two variables
 */
extern const char *epattern;
extern enum EmbeddingTranformation etransform;
extern enum Kernel kernel;

extern vector xformed_v;

#define KERNEL vdot
#define GET_CONTINOUS_VECTOR big_vector

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

            //debug("%d -> %d is correct\n", model[i], i);
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

PerceptronModel create_PerceptronModel(size_t transformed_embedding_length, IntegerIndexedFeatures iif) {

    PerceptronModel model = (PerceptronModel) malloc(sizeof (struct PerceptronModel));
    check_mem(model);

    model->embedding_w = NULL;
    model->discrete_w = NULL;

    model->c = 1;

    //FeatureMatrix mat = training->feature_matrix_singleton;

    if (transformed_embedding_length > 0) {
        model->embedding_w = vector_create(transformed_embedding_length);
        model->embedding_w_avg = vector_create(transformed_embedding_length);
        model->embedding_w_temp = vector_create(transformed_embedding_length);
        model->embedding_w_best = vector_create(transformed_embedding_length);

        log_info("Embedding features of %ld dimension", model->embedding_w->n);
    } else {
        model->embedding_w = NULL;
        model->embedding_w_avg = NULL;
        model->embedding_w_temp = NULL;
    }

    /*
    if (training->disrete_patterns_parts) {
        model->discrete_w = vector_create(iif->feature_id);
        model->discrete_w_avg = vector_create(iif->feature_id);
        model->discrete_w_temp = vector_create(iif->feature_id);

        log_info("Discrete features of %ld dimension", model->discrete_w->n);
    } else {
        model->discrete_w = NULL;
        model->discrete_w_avg = NULL;
        model->discrete_w_temp = NULL;

    }
     */

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
        vector_free(model->embedding_w_best);
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

struct Rate {
    double t_begin;
    double t_end;

    double total_elapsed;

    int count;
};

typedef struct Rate* Rate;

Rate parser_rate = NULL;

Rate start(Rate *r) {

    if (*r == NULL) {
        *r = (Rate) malloc(sizeof (struct Rate));
        (*r)->count = 0;
        (*r)->total_elapsed = 0.0;
        check(r != NULL, "Memory allocation error");
    }

    (*r)->t_begin = dsecnd();
    (*r)->t_end = -1;

    return *r;

error:
    exit(1);

}

void stop(Rate *r) {

    (*r)->t_end = dsecnd();
    (*r)->total_elapsed += ((*r)->t_end - (*r)->t_begin);
    ((*r)->count)++;

    if (((*r)->count) % PARSER_RATE_VERBOSITY == 0 && ((*r)->count) > 0) {
        log_info("Parser Rate is %lf sentences/sec", ((*r)->count) / ((*r)->total_elapsed));
    }
}

void train_once_PerceptronModel(PerceptronModel mdl, const CoNLLCorpus corpus, int max_rec) {
    long match = 0, total = 0;
    //size_t slen=0;

    log_info("Total number of training instances %d", (max_rec == -1) ? DArray_count(corpus->sentences) : max_rec);
    for (int si = 0; si < ((max_rec == -1) ? DArray_count(corpus->sentences) : max_rec); si++) {
        //log_info("Parsing sentence %d/%d", si+1, DArray_count(corpus));
        FeaturedSentence sent = (FeaturedSentence) DArray_get(corpus->sentences, si);


        start(&parser_rate);
        //debug("Building feature matrix for sentence %d of length %d", si, sent->length);
        //set_FeatureMatrix(NULL, corpus, si);


        //printfembedding(sent->feature_matrix, sent->length);

        debug("Building adjacency matrix for sentence %d of length %d", si, sent->length);
        build_adjacency_matrix(corpus, si, mdl->embedding_w, NULL);

        //printfmatrix(sent->adjacency_matrix, sent->length);

        //log_info("Adjacency matrix construction is done");


        int *model = parse(sent);
        stop(&parser_rate);
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

                if (model[i] != empirical[i]){
                    // -1 for Model arch
                    embedding_feature(sent, model[i], i, xformed_v);
                    vadd(mdl->embedding_w, xformed_v, -1.0);
                    vadd(mdl->embedding_w_temp, xformed_v, -1.0);
                    vadd(mdl->embedding_w_avg, xformed_v, -(mdl->c));

                    // +1 for Gold arc
                    embedding_feature(sent, empirical[i], i, xformed_v);

                    vadd(mdl->embedding_w, xformed_v, 1.0);
                    vadd(mdl->embedding_w_temp, xformed_v, 1.0);
                    vadd(mdl->embedding_w_avg, xformed_v, (mdl->c));
                }

                //free(real_embedding);
                //free(model_embedding);
            }
        }

        free_feature_matrix(corpus, si);

        mdl->c++;

        match += nm;
        total += (sent->length);

        if (si % 1000 == 0 && si > 0) {
            log_info("Running training accuracy %lf", (match * 1.) / total);

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
}

void dump_PerceptronModel(FILE *fp, int edimension, vector w, int best_numit) {

    fprintf(fp, "edimension=%d\n", edimension);
    fprintf(fp, "epattern=%s\n", epattern);
    fprintf(fp, "bestnumit=%d\n", best_numit);

    switch (etransform) {
        case LINEAR:
            fprintf(fp, "transformation=LINEAR\n");
            break;
        case QUADRATIC:
            fprintf(fp, "transformation=QUADRATIC\n");
            break;
    }

    fprintf(fp, "dimension=%ld\n", w->n);

    for (int i = 0; i < w->n; i++) {
        fprintf(fp, "%d=%f\n", i, w->data[i]);
    }
}

void train_perceptron_parser(PerceptronModel mdl, const CoNLLCorpus corpus, int numit, int max_rec) {

    for (int n = 0; n < numit; n++) {

        log_info("%d iteration in parser training...", (n + 1));

        train_once_PerceptronModel(mdl, corpus, max_rec);
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

HeadPredictionMetric create_HeadPredictionMetric() {
    HeadPredictionMetric hpm;

    hpm = (HeadPredictionMetric) malloc(sizeof (struct HeadPredictionMetric));

    check(hpm != NULL, "Memory allocation error for HeadPredictionMetric struct");

    hpm->total_prediction = 0;
    hpm->true_prediction = 0;



    return hpm;
error:

    exit(1);
}

void free_HeadPredictionMetric(HeadPredictionMetric hpm) {
    free(hpm);
}

ParserTestMetric create_ParserTestMetric() {
    ParserTestMetric ptm;

    ptm = (ParserTestMetric) malloc(sizeof (struct ParserTestMetric));

    check(ptm != NULL, "Memory allocation error on ParserTestMetric");

    ptm->all = create_HeadPredictionMetric();
    ptm->without_punc = create_HeadPredictionMetric();

    ptm->true_root_predicted = 0;
    ptm->total_sentence = 0;
    ptm->complete_sentence = 0;
    ptm->complete_sentence_without_punc = 0;

    return ptm;

error:
    exit(1);
}

void freeParserTestMetric(ParserTestMetric ptm) {
    free(ptm->all);
    free(ptm->without_punc);

    free(ptm);
}

void printParserTestMetric(ParserTestMetric metric) {

    log_info("\t Parent prediction accuracy: %f(%d out of %d)", (metric->all->true_prediction)*1. / (metric->all->total_prediction), metric->all->true_prediction, metric->all->total_prediction);
    log_info("\t Parent prediction accuracy (punctuations excluded): %f(%d out of %d)", (metric->without_punc->true_prediction)*1. / (metric->without_punc->total_prediction), metric->without_punc->true_prediction, metric->without_punc->total_prediction);

    log_info("\t ROOT Prediction accuracy: %f (%d out of %d)", (metric->true_root_predicted * 1.) / metric->total_sentence, metric->true_root_predicted, metric->total_sentence);

    log_info("\t Complete sentence: %f (%d out of %d)", (metric->complete_sentence * 1.) / metric->total_sentence, metric->complete_sentence, metric->total_sentence);
    log_info("\t Complete sentence (punctuations excluded): %f (%d out of %d)", (metric->complete_sentence_without_punc * 1.) / metric->total_sentence, metric->complete_sentence_without_punc, metric->total_sentence);
}

void mark_best_PerceptronModel(PerceptronModel model, int numit) {
    model->best_numit = numit;
    memcpy(model->embedding_w_best->data, model->embedding_w_temp->data, sizeof (float)*model->embedding_w_best->n);
}