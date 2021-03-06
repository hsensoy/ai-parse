//
//  corpus.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 13/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "corpus.h"
#include <dirent.h>
#include "stringalgo.h"

#include "debug.h"
#include "dependency.h"
#include "memman.h"
#include "conll.h"

#include <string.h>

#ifdef __GNUC__
#include <sys/types.h>
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

Word Root = NULL;

size_t max_num_sv = 0;
size_t max_narc = 0;

float *C = NULL, *r = NULL, *y = NULL;

vector xformed_v = NULL;
vector bigvector = NULL;

/**
 * ai-parse.c file for actual storage allocation for those two variables
 */
extern const char *epattern;
extern enum EmbeddingTranformation etransform;
extern float rbf_lambda;
extern int edimension;


DArray* embedding_pattern_parts = NULL;

EmbeddingPattern create_EmbeddingPattern() {
    EmbeddingPattern pattern = (EmbeddingPattern) malloc(sizeof (struct EmbeddingPattern));
    check(pattern != NULL, "Embedding Pattern allocation error");

    return pattern;

error:
    exit(1);
}

// Singleton

DArray* get_embedding_pattern_parts() {
    if (embedding_pattern_parts == NULL) {
        if (epattern != NULL) {
            DArray* patterns = split(epattern, "_");

            embedding_pattern_parts = DArray_create(sizeof (EmbeddingPattern), DArray_count(patterns));

            for (int pi = 0; pi < DArray_count(patterns); pi++) {
                char *pattern = (char*) DArray_get(patterns, pi);

                EmbeddingPattern ep = create_EmbeddingPattern();

                if (strcmp(pattern, "tl") == 0) { //thresholded-length
                    ep->node = 'l';
                    ep->subnode = 't';
                } else if (strcmp(pattern, "nl") == 0) { // normalized-length
                    ep->node = 'l';
                    ep->subnode = 'n';
                } else if (strcmp(pattern, "l") == 0) { // raw length
                    ep->node = 'l';
                    ep->subnode = 'r';
                } else if (strcmp(pattern, "lbf") == 0) { // Left Boundary Flag
                    ep->node = 'b';
                    ep->subnode = 'l';
                } else if (strcmp(pattern, "rbf") == 0) { // Right Boundary Flag
                    ep->node = 'b';
                    ep->subnode = 'r';
                } else if (strcmp(pattern, "dir") == 0) { // Direction
                    ep->node = 'd';
                } else if (strcmp(pattern, "root") == 0) { // Root Flag
                    ep->node = 'r';
                } else if (strcmp(pattern, "between") == 0) { //Between words
                    ep->node = 'w';
                } else {

                    int n = sscanf(pattern, "%c%dv", &(ep->node), &(ep->offset));

                    check(n == 2, "Expected pattern format is [p|c]<offset>v where as got %s", pattern);
                    check(ep->node == 'p' || ep->node == 'c', "Unknown node name %c expected p or c", ep->node);
                }

                DArray_push(embedding_pattern_parts, ep);
            }

            debug("Number of embedding patterns is %d", DArray_count(embedding_pattern_parts));

        } else {
            embedding_pattern_parts = NULL;
        }

        return embedding_pattern_parts;
    } else {
        return embedding_pattern_parts;
    }

error:
    return NULL;
}

Word ROOT(int dim) {
    if (Root == NULL) {
        Root = (Word) malloc(sizeof (struct Word));
        check_mem(Root);

        Root->id = 0;
        Root->form = strdup("ROOT");
        Root->postag = strdup("ROOT");
        Root->parent = -1;


        Root->embedding = vector_create(dim);

    }


    return Root;


error:

    exit(1);
}

/**
 * 
 * @param base_dir CoNLL base directory including sections
 * @param sections DArray of sections
 * @param embedding_dimension Embedding dimension per word
 * @param discrete_patterns Reserved for future use
 * @return CoNLLCorpus structure
 */
CoNLLCorpus create_CoNLLCorpus(const char* base_dir, DArray *sections, int embedding_dimension, DArray* discrete_patterns) {
    CoNLLCorpus corpus = (CoNLLCorpus) malloc(sizeof (struct CoNLLCorpus));

    check_mem(corpus);

    corpus->base_dir = base_dir;
    corpus->sections = sections;

    corpus->sentences = DArray_create(sizeof (FeaturedSentence), 2000);
    check_mem(corpus->sentences);

    corpus->hasembeddings = embedding_dimension > 0;

    if (discrete_patterns) {
        corpus->disrete_patterns_parts = DArray_create(sizeof (DArray*), DArray_count(discrete_patterns));
        check_mem(corpus->disrete_patterns_parts);

        for (int i = 0; i < DArray_count(discrete_patterns); i++)
            DArray_push(DArray_get(corpus->disrete_patterns_parts, i), split(((char*) DArray_get(discrete_patterns, i)), "_"));


    } else
        corpus->disrete_patterns_parts = NULL;

    corpus->Root = ROOT(embedding_dimension);
    corpus->word_embedding_dimension = embedding_dimension;
    corpus->transformed_embedding_length = -1;

    int embedding_concat_length = 1;
    for (int pi = 0; pi < DArray_count(get_embedding_pattern_parts()); pi++) {
        EmbeddingPattern pattern = (EmbeddingPattern) DArray_get(get_embedding_pattern_parts(), pi);

        if (pattern->node == 'p' || pattern->node == 'c' || pattern->node == 'w')
            embedding_concat_length += embedding_dimension;
        else if (pattern->node == 'l' && (pattern->subnode == 'r' || pattern->subnode == 'n'))
            embedding_concat_length += 1;
        else if (pattern->node == 'l' && pattern->subnode == 't')
            embedding_concat_length += 9;
        else if (pattern->node == 'b')
            embedding_concat_length += 2;
        else if (pattern->node == 'r')
            embedding_concat_length += 1;
        else if (pattern->node == 'd')
            embedding_concat_length += 2;
    }

    // Be optimistic about LINEAR transformation
    bigvector = vector_create(embedding_concat_length);
    corpus->transformed_embedding_length = embedding_concat_length;
    if (etransform == QUADRATIC)
        corpus->transformed_embedding_length = ((corpus->transformed_embedding_length) * (corpus->transformed_embedding_length + 1)) / 2;
    else if (etransform == CUBIC) {
        size_t emprical_xform_length = 0;
        for (size_t i = 0; i < embedding_concat_length; i++) {
            for (size_t j = 0; j <= i; j++) {
                for (size_t k = 0; k <= j; k++) {
                    emprical_xform_length++;
                }
            }
        }

        corpus->transformed_embedding_length = emprical_xform_length;
    }


    xformed_v = vector_create(corpus->transformed_embedding_length);


    log_info("Corpus has an embedding length of %d (%ld by %d transformation)", embedding_dimension, corpus->transformed_embedding_length, etransform);



    return corpus;
error:
    exit(1);
}

void free_CoNLLCorpus(CoNLLCorpus corpus, bool free_feature_matrix) {

    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        //FeaturedSentence sent = (FeaturedSentence)DArray_get(sentences, i);

        free_FeaturedSentence(corpus, si);
    }

    debug("Sentences are released");

    if (corpus->disrete_patterns_parts != NULL)
        DArray_clear_destroy(corpus->disrete_patterns_parts);

    DArray *epparts = get_embedding_pattern_parts();
    if (epparts != NULL)
        DArray_clear_destroy(epparts);

    debug("Patterns are released");


}

/**
 * 
 * @param has_discrete_features 
 * @param embedding_length      Length of the embedding vector to be used.
 * @return     Creates a feature vector as union of discrete and continuous vectors.
 */
FeatureVector FeatureVector_create(bool has_discrete_features, uint32_t embedding_length) {
    FeatureVector fv = (FeatureVector) malloc(sizeof (struct FeatureVector));

    check(fv != NULL, "Error in allocating a FeatureVector");

    if (has_discrete_features) {
        fv->discrete_v = DArray_create(sizeof (uint32_t), 18);
        check_mem(fv->discrete_v);
    } else {
        fv->discrete_v = NULL;
    }

    if (embedding_length > 0) {
        fv->continous_v = vector_create(embedding_length);
    } else {
        fv->continous_v = NULL;
    }

    return fv;
error:
    exit(1);
}

void free_FeatureVector(FeatureVector v) {

    //DArray_destroy(v->discrete_v);

    vector_free(v->continous_v);

    free(v);

}

FeatureMatrix FeatureMatrix_create(int sent_length, uint32_t embedding_length, bool has_discrete_features) {
    FeatureMatrix matrix = (FeatureMatrix) malloc(sizeof (struct FeatureMatrix));

    check(matrix != NULL, "Error in allocating matrix FeatureMatrix");

    matrix->size = sent_length + 1;

    matrix->matrix_data = (FeatureVector**) malloc(sizeof (FeatureVector*) * (matrix->size));
    check(matrix->matrix_data != NULL, "Error in allocating 2-dimensional FeatureVector");
    matrix->embedding_length = embedding_length;
    matrix->has_discrete_features = has_discrete_features;

    debug("Embedding vector length is %ld", embedding_length);

    log_info("Allocating a %d x %d FeatureMatrix", matrix->size, matrix->size);
    for (int i = 0; i < matrix->size; i++) {
        (matrix->matrix_data)[i] = (FeatureVector*) malloc(sizeof (FeatureVector) * (matrix->size));

        check((matrix->matrix_data)[i] != NULL, "Error in allocating FeatureVector row");

        for (int j = 0; j < matrix->size; j++) {
            if (i == j || j == 0)
                (matrix->matrix_data)[i][j] = NULL;
            else {
                (matrix->matrix_data)[i][j] = FeatureVector_create(has_discrete_features, embedding_length);
            }
        }
    }

    return matrix;

error:
    exit(1);
}

void free_featureMatrix(FeatureMatrix matrix) {
    for (int i = 0; i < matrix->size; i++) {

        for (int j = 0; j < matrix->size; j++) {
            if (i == j || j == 0)
                (matrix->matrix_data)[i][j] = NULL;
            else {
                free_FeatureVector((matrix->matrix_data)[i][j]);
            }
        }

        free((matrix->matrix_data)[i]);
    }

    free(matrix->matrix_data);
    free(matrix);
}

/**
 * 
 * @param sent
 * @param from
 * @param to
 * @param target When NULL a new vector is created by vlinear/vquadratic functions. Release of memory is deferred to user.
 *                      When a non-NULL vector is given vlinear/vquadratic functions simply perform a copy operation with no new allocation.
 * @return 
 */
vector embedding_feature(FeaturedSentence sent, int from, int to, vector target) {
    bigvector->last_idx = 0;
    IS_ARC_VALID(from, to, sent->length);

    DArray* patterns = get_embedding_pattern_parts();

    for (int pi = 0; pi < DArray_count(patterns); pi++) {
        EmbeddingPattern pattern = (EmbeddingPattern) DArray_get(patterns, pi);

        if (pattern->node == 'p') {

            if (from == 0)
                bigvector = vconcat(bigvector, Root->embedding);
            else if (from + pattern->offset >= 1 && from + pattern->offset <= sent->length) {
                bigvector = vconcat(bigvector, ((Word) DArray_get(sent->words, from + pattern->offset - 1))->embedding);
            } else {
                bigvector = vconcat(bigvector, Root->embedding);
            }
        } else if (pattern->node == 'c') {

            if (to + pattern->offset >= 1 && to + pattern->offset <= sent->length) {
                bigvector = vconcat(bigvector, ((Word) DArray_get(sent->words, to + pattern->offset - 1))->embedding);
            } else {
                bigvector = vconcat(bigvector, Root->embedding);
            }


        } else if (pattern->node == 'w') {

            //log_info("Embedding dimension %d",edimension);
            vector avg_v = vector_create(edimension);

            for (size_t i = 0; i < avg_v->n; i++)
                (avg_v->data)[i] = 0.0;

            //log_info("Initialization is done");

            if (abs(from - to) > 1) {

                int n = 0;

                for (int b = MIN(from, to) + 1; b < MAX(from, to); b++) {


                    //log_info("from=%d, to=%d, b=%d",MIN(from, to),MAX(from, to), b);
                    vector b_vec = ((Word) DArray_get(sent->words, b - 1))->embedding;

                    for (size_t bi = 0; bi < b_vec->n; bi++)
                        (avg_v->data)[bi] += (b_vec->data)[bi];


                    n++;
                }

                for (size_t bi = 0; bi < avg_v->n; bi++)
                    (avg_v->data)[bi] /= n;

            }


            bigvector = vconcat(bigvector, avg_v);
            vector_free(avg_v);

        } else if (pattern->node == 'l') {

            if (pattern->subnode == 't') {
                const int threshold_arr[] = {1, 2, 3, 4, 5, 10, 20, 30, 40};
                float threshold_flag[9];

                for (int i = 0; i < 9; i++)
                    if (abs(from - to) > threshold_arr[i])
                        threshold_flag[i] = 1;
                    else
                        threshold_flag[i] = 0;

                bigvector = vconcat_arr(bigvector, 9, threshold_flag);

            } else if (pattern->subnode == 'r') {
                vector length_v = vector_create(1);

                length_v->data[0] = abs(from - to);

                bigvector = vconcat(bigvector, length_v);

                vector_free(length_v);
            } else if (pattern->subnode == 'n') {

                vector length_v = vector_create(1);

                // TODO: Are you an idiot ?
                length_v->data[0] = abs(from - to) / 250.;

                bigvector = vconcat(bigvector, length_v);

                vector_free(length_v);

            }

        } else if (pattern->node == 'b') {
            if (pattern->subnode == 'l') {
                float left_boundary[] = {0, 0};

                left_boundary[0] = 0;
                left_boundary[1] = 0;


                if (from == 1)
                    left_boundary[0] = 1;

                if (to == 1)
                    left_boundary[1] = 1;


                bigvector = vconcat_arr(bigvector, 2, left_boundary);

            } else if (pattern->subnode == 'r') {
                float right_boundary[] = {0, 0};

                right_boundary[0] = 0;
                right_boundary[1] = 0;


                if (from == sent->length)
                    right_boundary[0] = 1;

                if (to == sent->length)
                    right_boundary[1] = 1;

                bigvector = vconcat_arr(bigvector, 2, right_boundary);
            }
        } else if (pattern->node == 'r') {
            const float root_true[] = {1.};
            const float root_false[] = {0.};

            if (from == 0)
                vconcat_arr(bigvector, 1, root_true);
            else
                vconcat_arr(bigvector, 1, root_false);


        } else if (pattern->node == 'd') {
            const float left2right[] = {1., 0.};
            const float right2left[] = {0., 1.};

            if (from < to) {
                vconcat_arr(bigvector, 2, left2right);
            } else {
                vconcat_arr(bigvector, 2, right2left);
            }
        }
    }


    // Add the bias term
    const float bias[] = {1.};
    bigvector = vconcat_arr(bigvector, 1, bias);


    switch (etransform) {
        case LINEAR:
            return vlinear(target, bigvector);
            break;
        case QUADRATIC:
            return vquadratic(target, bigvector, 1);
            break;
        case CUBIC:
            return vcubic(target, bigvector, target->n);
            break;
    }

error:
    return NULL;
}

void build_embedding_feature(FeaturedSentence sent, int from, int to) {
    embedding_feature(sent, from, to, ((sent->feature_matrix_ref->matrix_data)[from][to])->continous_v);
}

void set_FeatureMatrix(Hashmap* featuremap, CoNLLCorpus corpus, int sentence_idx) {

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    //check(corpus->feature_matrix_singleton->size >= length, "Singleton Matrix is too small (%d) for a sentence length of (%d). Fix and recompile the code", corpus->feature_matrix_singleton->size, length);

    for (int _from = 0; _from <= length; _from++)
        for (int _to = 1; _to <= length; _to++) {
            if (_to != _from) {
                if (corpus->disrete_patterns_parts)
                    (sentence->feature_matrix_ref->matrix_data)[_from][_to]->discrete_v = NULL;

                if (corpus->hasembeddings) {
                    build_embedding_feature(sentence, _from, _to);



                    //if ((sentence->feature_matrix)[_from][_to]->continous_v->true_n > 50)
                    //	log_info("%ld",(sentence->feature_matrix)[_from][_to]->continous_v->true_n);
                }

            }
        }

    return;
}

void free_feature_matrix(CoNLLCorpus corpus, int sentence_idx) {
    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    //int length = sentence->length;

    sentence->feature_matrix_ref = NULL;
}

float** square_adjacency_matrix(int n, float init_value) {

    float** matrix = (float**) malloc(sizeof (float*) * n);
    check_mem(matrix);
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*) malloc(sizeof (float) * n);

        for (int j = 0; j < n; j++) {

            if (i == j)
                matrix[i][j] = init_value;
            else
                matrix[i][j] = 0.0;
        }

        check_mem(matrix[i]);
    }

    return matrix;
error:
    log_err("adjacency_matrix allocation error");
    exit(1);
}

float* get_embedding_matrix(CoNLLCorpus corpus, int sentence_idx, size_t *m, size_t *n) {
    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    *m = (length + 1) * length - length;
    *n = (sentence->feature_matrix_ref->matrix_data)[0][1]->continous_v->n;

    debug("Embedding matrix is %d x %d", *m, *n);

    float *matrix = (float*) mkl_64bytes_malloc((*m) * (*n) * sizeof (float));

    int offset = 0;
    for (int _from = 0; _from <= length; _from++) {
        for (int _to = 1; _to <= length; _to++) {
            if (_to != _from) {

                vector embedding = (sentence->feature_matrix_ref->matrix_data)[_from][_to]->continous_v;

                for (int i = 0; i < embedding->n; i++)
                    matrix[offset++] = (embedding->data)[i];
            }
        }
    }


    check(offset == (*m) * (*n), "Matrix is not of the same size with the embeddings dimension x # of support vectors");

    return matrix;

error:
    exit(1);
}

void set_adj_matrix_mkl(CoNLLCorpus corpus, int sentence_idx, const float* y) {
    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    int offset = 0;

    for (int _from = 0; _from <= length; _from++) {
        for (int _to = 1; _to <= length; _to++)
            if (_to != _from)
                (sentence->adjacency_matrix)[_from][_to] = y[offset++];
    }

    check(offset == (length + 1) * length - length, "Matrix is not of the same size with the embeddings dimension x # of support vectors");

    return;
error:
    exit(1);
}

void set_adjacency_matrix_fast(CoNLLCorpus corpus, int sentence_idx, KernelPerceptron kp, bool use_avg_alpha) {

    size_t num_sv, narc, edim;
    float* embedding_matrix;



    num_sv = kp->M;

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    if (sentence->adjacency_matrix == NULL)
        sentence->adjacency_matrix = square_adjacency_matrix(length + 1, NEGATIVE_INFINITY);


    if (num_sv > 0) {
        embedding_matrix = get_embedding_matrix(corpus, sentence_idx, &narc, &edim);

        bool narc_changed = false, num_sv_changed = false;
        if (narc > max_narc) {
            max_narc = narc + 4;
            narc_changed = true;
        }

        if (num_sv > max_num_sv) {
            max_num_sv = num_sv + 2048;
            num_sv_changed = true;
        }

        if (num_sv_changed || narc_changed) {
            log_info("REALLOC: C(%lu) and r(%lu)", max_num_sv, max_narc);
            C = (float*) mkl_64bytes_realloc(C, max_num_sv * max_narc * sizeof (float));
            r = (float*) mkl_64bytes_realloc(r, max_num_sv * max_narc * sizeof (float));
        }

        if (narc_changed) {
            log_info("REALLOC: y(%lu)", max_narc);
            y = (float*) mkl_64bytes_realloc(y, max_narc * sizeof (float));
        }

        if (kp->kernel == KPOLYNOMIAL) {
#pragma vector nontemporal (C, r)
#pragma loop_count min(30000), max(640000000), avg(1000000)
#pragma ivdep
            for (size_t i = 0; i < num_sv * narc; i++) {
                C[i] = kp->bias;
                //r[i] = 0.;
            }

            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
            //        narc, num_sv, edim, 1., embedding_matrix, edim, kp->kernel_matrix, num_sv, 1, C,num_sv);


            debug("Matrix multiplication");
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    narc, num_sv, edim, 1., embedding_matrix, edim, kp->kernel_matrix, edim, 1, C, num_sv);


            debug("Power it");

            check(num_sv * narc > 0, "num_sv=%lu and narc=%lu is not valid. Check your code", num_sv, narc);
#pragma loop_count min(3000), max(640000000), avg(1000000)
#pragma ivdep
            for (size_t i = 0; i < num_sv * narc; i++)
                r[i] = pow(C[i], kp->power);

            //vsPowx(num_sv*narc, C, kp->power, r);

            debug("Matrix vector mult");
            if (use_avg_alpha)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, narc, num_sv, 1., r, num_sv, kp->alpha_avg, 1, 0., y, 1);
            else
                cblas_sgemv(CblasRowMajor, CblasNoTrans, narc, num_sv, 1., r, num_sv, kp->alpha, 1, 0., y, 1);

        } else if (kp->kernel == KRBF) {

            float *delta = (float*) mkl_64bytes_malloc(edim * sizeof (float));

            for (size_t i = 0; i < narc; i++) {

                float *varc = embedding_matrix + i * edim;


                y[i] = 0.0;
                for (size_t isv = 0; isv < num_sv; isv++) {

                    float *sv = kp->kernel_matrix + isv * edim;



                    vsSub(edim, sv, varc, delta);

                    if (use_avg_alpha)
                        y[i] += kp->alpha_avg[isv] * exp(-kp->rbf_lambda * pow(cblas_snrm2(edim, delta, 1), 2));
                    else
                        y[i] += kp->alpha[isv] * exp(-kp->rbf_lambda * pow(cblas_snrm2(edim, delta, 1), 2));
                }
            }

            mkl_free(delta);

        } else if (kp->kernel == KLINEAR) {
            log_err("Linear kernel is not implemented yet");
            exit(1);
        }

        debug("Set adjacency");
        set_adj_matrix_mkl(corpus, sentence_idx, y);


        mkl_free(embedding_matrix);
    }

    return;
error:
    exit(1);
}

void build_adjacency_matrix(CoNLLCorpus corpus, int sentence_idx, vector embeddings_w, vector discrete_w) {

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    if (sentence->adjacency_matrix == NULL)
        sentence->adjacency_matrix = square_adjacency_matrix(length + 1, NEGATIVE_INFINITY);


    //sentence->feature_matrix = FeatureMatrix_create(length, corpus->hasembeddings, corpus->disrete_patterns_parts != NULL);

    for (int _from = 0; _from <= length; _from++)
        for (int _to = 1; _to <= length; _to++) {
            if (_to != _from) {
                (sentence->adjacency_matrix)[_from][_to] = 0.0;
                if (corpus->disrete_patterns_parts)
                    (sentence->adjacency_matrix)[_from][_to] = -1; // TODO: Complete discrete dot product.

                if (corpus->hasembeddings) {

                    //debug("%d->%d\n", _from, _to);
                    xformed_v = embedding_feature(sentence, _from, _to, xformed_v);

                    if (xformed_v == NULL) {
                        log_err("NULL continous vector");
                        exit(EXIT_FAILURE);
                    }

                    //vprint(embedding);
                    (sentence->adjacency_matrix)[_from][_to] += vdot(embeddings_w, xformed_v);

                }

            }
        }
}

Word parse_word(char* line, int embedding_dimension) {
    Word w = (Word) malloc(sizeof (struct Word));
    check_mem(w);

    w->conll_piece = split(line, "\t");

    w->id = atoi((char*) DArray_get(w->conll_piece, 0));
    //free((char*) DArray_get(tokens, 0));

    w->form = (char*) DArray_get(w->conll_piece, 1);
    w->postag = (char*) DArray_get(w->conll_piece, 3);

    w->parent = atoi((char*) DArray_get(w->conll_piece, 6));
    //free((char*) DArray_get(tokens, 6));

    if (embedding_dimension > 0) {
        check(DArray_count(w->conll_piece) >= 11, "CoNLL files in corpus with embedding should contain at least 11 fields. 11. field being the embedding field. Found a line with only %d fields", DArray_count(w->conll_piece));

        w->embedding = parse_vector((char*) DArray_get(w->conll_piece, 10));
        //free((char*) DArray_get(tokens, 10));

        check(embedding_dimension == w->embedding->n, "Expected embedding dimension was %d but got %ld", embedding_dimension, w->embedding->n);
    } else
        w->embedding = NULL;

    return w;

error:
    exit(1);
}

void Word_free(Word w) {
    vector_free(w->embedding);

    DArray_clear_destroy(w->conll_piece);

    free(w);
}

void add_word(FeaturedSentence sentence, Word word) {

    DArray_push(sentence->words, word);

    sentence->length++;
}

FeaturedSentence FeatureSentence_create() {

    FeaturedSentence sent = (FeaturedSentence) malloc(sizeof (struct FeaturedSentence));
    check_mem(sent);

    sent->words = DArray_create(sizeof (Word), 10);
    check_mem(sent->words);

    sent->length = 0;
    sent->feature_matrix_ref = NULL;
    sent->adjacency_matrix = NULL;

    return sent;

error:
    log_err("Sentence allocation error.");
    exit(1);
}



// TODO: Complete implementation

void free_FeaturedSentence(CoNLLCorpus corpus, int sentence_idx) {

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);

    for (int wi = 0; wi < DArray_count(sentence->words); wi++) {

        Word word = (Word) DArray_get(sentence->words, wi);

        Word_free(word);
    }

}

static DArray* find_corpus_files(const char *dir, DArray* sections) {
    struct dirent *entry;
    DIR *dp;

    DArray *array = DArray_create(sizeof (char*), 100);

    check(array != NULL, "Corpus file array creation failed.");

    char path[255];
    for (int i = 0; i < DArray_count(sections); i++) {
        int section = *((int*) DArray_get(sections, i));
        sprintf(path, "%s/%02d", dir, section);

        dp = opendir(path);
        check(dp != NULL, "Directory access error %s", path);

        while ((entry = readdir(dp))) {
            if (endswith(entry->d_name, ".dp")) {

                conll_file_t file = create_CoNLLFile(dir, section, entry->d_name);

                DArray_push(array, file);
            }
        }

        closedir(dp);
    }

    return array;

error:
    log_err("Terminating...");
    exit(1);
}

void read_corpus(CoNLLCorpus corpus, bool build_feat_matrix) {
    DArray* files = find_corpus_files(corpus->base_dir, corpus->sections);

    char *line = NULL;
    size_t len = 0;

    FeaturedSentence sent = FeatureSentence_create();

    for (int i = 0; i < DArray_count(files); i++) {
        ssize_t read;
        conll_file_t file = (conll_file_t) DArray_get(files, i);

        FILE *fp = fopen(file->fullpath, "r");
        check_mem(fp);

        while ((read = getline(&line, &len, fp)) != -1) {

            if (strcmp(line, "\n") != 0) {
                Word w = parse_word(line, corpus->word_embedding_dimension);

                add_word(sent, w);

            } else {
                sent->section = file->section;
                DArray_push(corpus->sentences, sent);

                //debug("One more sentence is added into corpus...");

                if (build_feat_matrix)
                    set_FeatureMatrix(NULL, corpus, DArray_count(corpus->sentences) - 1);

                sent = FeatureSentence_create();
            }

        }

        fclose(fp);
    }

    free(line);

    // DArray_clear_destroy(files);

    log_info("Total of %d sentences", DArray_count(corpus->sentences));


    return;
error:
    log_err("Terminating...");
    exit(1);

}

