#include "perceptron.h"
#include "debug.h"
#include "uthash.h"
#include "corpus.h"
#include "dependency.h"
#include "memman.h"
#include "parseutil.h"

extern enum BudgetMethod budget_method;
extern size_t budget_target;

extern int verbosity;
extern const char *modelname;
extern enum Kernel kernel;

#include <time.h>
#include <stdlib.h>

KernelPerceptron create_PolynomialKernelPerceptron(int power, float bias) {

    srand(time(NULL));

    KernelPerceptron kp = (KernelPerceptron) malloc(sizeof (struct KernelPerceptron));

    check(kp != NULL, "KernelPerceptron allocation error");

    kp->M = 0;
    kp->alpha = NULL;
    kp->beta = NULL;
    kp->alpha_avg = NULL;
    kp->best_alpha_avg = NULL;
    kp->best_kernel_matrix = NULL;
    kp->c = 1;
    kp->kernel_matrix = NULL;
    kp->arch_to_index_map = NULL;
    kp->kernel = KPOLYNOMIAL;
    kp->bias = bias;
    kp->power = power;

    log_info("Polynomial kernel of degree %d with bias %f is created", kp->power, kp->bias);


    return kp;


error:
    exit(1);

}

KernelPerceptron create_RBFKernelPerceptron(float lambda) {

    srand(time(NULL));

    KernelPerceptron kp = (KernelPerceptron) malloc(sizeof (struct KernelPerceptron));

    check(kp != NULL, "KernelPerceptron allocation error");

    kp->M = 0;
    kp->alpha = NULL;
    kp->beta = NULL;
    kp->alpha_avg = NULL;
    kp->best_alpha_avg = NULL;
    kp->best_kernel_matrix = NULL;
    kp->c = 1;
    kp->kernel_matrix = NULL;
    kp->arch_to_index_map = NULL;
    kp->kernel = KRBF;
    kp->rbf_lambda = lambda;

    log_info("RBF kernel with lambda %f is created", kp->rbf_lambda);


    return kp;


error:
    exit(1);

}

/*
alpha_t* create_alpha_idx(int sentence_idx, int from, int to, int length) {

    IS_ARC_VALID(from, to, length);

    alpha_t *idx = (alpha_t*) malloc(sizeof (alpha_t));

    check_mem(idx);

    idx->sentence_idx = sentence_idx;
    idx->from = from;
    idx->to = to;

    return idx;

error:
    log_err("Error in allocating alpha index for sentence %d arch %d->%d", sentence_idx, from, to);
    exit(1);
}
 */

/**
 * Allocate and initialize an new alpha_key
 * 
 * @param sidx Sentence index
 * @param from from word index
 * @param to to word index
 * 
 * @return Initialized point to alpha_key allocated
 */
alpha_key_t* create_alpha_key(uint32_t sidx, uint16_t from, uint16_t to) {
    alpha_key_t *key;

    key = (alpha_key_t*) malloc(sizeof (alpha_key_t));
    check(key != NULL, "Alpha key allocation error");

    memset(key, 0, sizeof (alpha_key_t));

    key->sentence_idx = sidx;
    key->from = from;
    key->to = to;

    return key;

error:
    exit(1);
}

unsigned get_keylen() {
    return offsetof(alpha_t, to) /* offset of last key field */
            + sizeof (uint16_t) /* size of last key field */
            - offsetof(alpha_t, sentence_idx); /* offset of first key field */
}

void update_alpha(KernelPerceptron kp, uint32_t sidx, uint16_t from, uint16_t to, struct FeaturedSentence* sent, float inc) {

    unsigned keylen;
    alpha_t *a, *dummy = NULL;
    alpha_key_t *a_key = NULL;

    keylen = get_keylen();

    a_key = create_alpha_key(sidx, from, to);

    HASH_FIND(hh, kp->arch_to_index_map, &a_key->sentence_idx, keylen, a);
    if (a != NULL) {
        (kp->alpha)[a->idx] += inc;

        (kp->beta)[a->idx] += inc * kp->c;

    } else {
        vector v = embedding_feature(sent, from, to, NULL);
        size_t n = (kp->M);



        if (n == 0) {
            kp->N = v->n;
            kp->kernel_matrix = (float*) mkl_64bytes_malloc((n + 1) * v->n * sizeof (float));

            for (size_t i = 0; i < v->n; i++)
                (kp->kernel_matrix)[n * v->n + i] = v->data[i];

            kp->alpha = (float*) mkl_64bytes_malloc((n + 1) * sizeof (float));
            kp->beta = (float*) mkl_64bytes_malloc((n + 1) * sizeof (float));
            (kp->alpha)[n] = inc;
            (kp->beta)[n] = inc * kp->c;
        } else {
            check(kp->N == v->n, "%lu dimensional embedding does not confirm with the previous embedding size (%lu)", v->n, kp->N);

            kp->kernel_matrix = (float*) mkl_64bytes_realloc(kp->kernel_matrix, (n + 1) * v->n * sizeof (float));

            for (size_t i = 0; i < v->n; i++)
                (kp->kernel_matrix)[n * v->n + i] = v->data[i];

            kp->alpha = (float*) mkl_64bytes_realloc(kp->alpha, (n + 1) * sizeof (float));
            kp->beta = (float*) mkl_64bytes_realloc(kp->beta, (n + 1) * sizeof (float));
            (kp->alpha)[n] = inc;
            (kp->beta)[n] = inc * kp->c;
        }

        vector_free(v);

        a = (alpha_t*) malloc(sizeof (alpha_t));
        memset(a, 0, sizeof (alpha_t)); /* zero fill */

        a->from = from;
        a->to = to;
        a->sentence_idx = sidx;
        a->idx = n;


        HASH_ADD(hh, kp->arch_to_index_map, sentence_idx, keylen, a);


        debug("%u keys in alpha", HASH_COUNT(kp->arch_to_index_map));

        (kp->M)++;

    }

    return;
error:
    exit(1);
}

size_t delete_hypothesis(KernelPerceptron kp, size_t idx) {

    if (kp->M > 1) {

        unsigned keylen = get_keylen();

        alpha_t *a = NULL;
        alpha_t *rm_v = NULL, *mv_v = NULL;
        alpha_key_t *rm_k = NULL, *mv_k = NULL;


        for (a = kp->arch_to_index_map; a != NULL && (mv_v == NULL || rm_v == NULL); a = a->hh.next) {

            // You found the one to be moved 
            if (a->idx == kp->M - 1) {

                mv_v = a;
                //mv_k = create_alpha_key(a->sentence_idx, a->from,a->to);

            }

            // You found the one to be deleted
            if (a->idx == idx) {

                rm_v = a;

                //rm_k = create_alpha_key(a->sentence_idx, a->from,a->to);

            }

        }

        check(mv_v != NULL && rm_v != NULL, "Check the code. move_key or rm_key is NULL");

        int alpha_chopped = (int) (kp->alpha)[rm_v->idx];

        if (alpha_chopped != 1 && alpha_chopped != -1)
            return false;

        (kp->alpha)[rm_v->idx] = (kp->alpha)[mv_v->idx];
        kp->alpha = (float*) mkl_64bytes_realloc(kp->alpha, ((kp->M) - 1) * sizeof (float));

        (kp->beta)[rm_v->idx] = (kp->beta)[mv_v->idx];
        kp->beta = (float*) mkl_64bytes_realloc(kp->beta, ((kp->M) - 1) * sizeof (float));

        for (size_t i = 0; i < kp->N; i++) {

            kp->kernel_matrix[ rm_v->idx * kp->N + i ] = kp->kernel_matrix[ mv_v->idx * kp->N + i ];

        }

        kp->kernel_matrix = (float*) mkl_64bytes_realloc(kp->kernel_matrix, ((kp->M - 1) * kp->N) * sizeof (float));

        if (mv_v == rm_v) {
            HASH_DEL(kp->arch_to_index_map, mv_v);
        } else {

            HASH_DEL(kp->arch_to_index_map, mv_v);
            HASH_DEL(kp->arch_to_index_map, rm_v);
        }


        a = (alpha_t*) malloc(sizeof (alpha_t));
        memset(a, 0, sizeof (alpha_t)); /* zero fill */

        a->from = mv_v->from;
        a->to = mv_v->to;
        a->sentence_idx = mv_v->sentence_idx;
        a->idx = rm_v->idx;


        HASH_ADD(hh, kp->arch_to_index_map, sentence_idx, keylen, a);

        if (mv_v == rm_v) {
            free(mv_v);
        } else {
            free(mv_v);
            free(rm_v);
        }

        (kp->M)--;

        return true;

    } else {
        log_warn("There are only %lu hypothesis vectors left. Do deletion will be done.", kp->M);

        return false;
    }

error:
    exit(1);
}

#define MAX_FAIL_TO_DELETE 5

size_t delete_n_random_hypothesis(KernelPerceptron kp, size_t delete_count) {

    size_t ndel_success = 0;

    if (kp->M > delete_count + 1) {

        for (size_t i = 0; i < delete_count; i++) {

            bool deleted = false;
            int fail_to_delete = 0;

            while (!deleted && fail_to_delete < MAX_FAIL_TO_DELETE) {
                size_t r = ((size_t) rand()) % (kp->M);

                bool rc = delete_hypothesis(kp, r);

                if (rc) {
                    deleted = true;
                    ndel_success++;
                } else {
                    fail_to_delete++;
                }
            }

        }

    } else {
        log_warn("There are only %lu hypothesis vectors left. %lu vectors are asked for deletion.Do deletion will be done.", kp->M, delete_count);
    }

    return ndel_success;

}

void update_average_alpha(KernelPerceptron kp) {

    if (kp->alpha_avg != NULL) {
        mkl_free(kp->alpha_avg);
    }

    kp->alpha_avg = (float*) mkl_64bytes_malloc((kp->M) * sizeof (float));

    for (size_t i = 0; i < (kp->M); i++) {

        (kp->alpha_avg)[i] = (kp->alpha)[i] - (kp->beta)[i] / (kp->c);

    }

}

static void dump_support_vectors(KernelPerceptron mdl) {
    char filename[1024];
    time_t result = time(NULL);

    sprintf(filename, "%s.%d.sv", modelname, (int) result);


    FILE *fp = fopen(filename, "w");


    for (size_t i = 0; i < mdl->M; i++) {

        fprintf(fp, "%f\t", mdl->alpha[i]);
        for (size_t j = 0; j < mdl->N; j++) {

            fprintf(fp, "%f", mdl->kernel_matrix[i * mdl->N + j]);

            if (j < mdl->N - 1)
                fprintf(fp, "\t");



        }
        fprintf(fp, "\n");


    }

    fclose(fp);



}

void train_once_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, int max_rec) {
    long match = 0, total = 0;
    //size_t slen=0;

    double s_initial = dsecnd();
    int max_sv = 0;


    log_info("Total number of training instances %d", (max_rec == -1) ? DArray_count(corpus->sentences) : max_rec);

    for (int si = 0; si < ((max_rec == -1) ? DArray_count(corpus->sentences) : max_rec); si++) {

        FeaturedSentence sent = (FeaturedSentence) DArray_get(corpus->sentences, si);

        debug("Building feature matrix for sentence %d", si);
        set_FeatureMatrix(NULL, corpus, si);

        set_adjacency_matrix_fast(corpus, si, mdl, false);

        max_sv += (sent->length + 1) * sent->length - sent->length;

        int *model = parse(sent);

        //printfarch(model, sent->length);
        debug("Parsing sentence %d of length %d is done", si, sent->length);
        int *empirical = get_parents(sent);

        //printfarch(empirical, sent->length);
        int nm = nmatch(model, empirical, sent->length);

        debug("Model matches %d arcs out of %d arcs", nm, sent->length);
        if (nm != sent->length) { // root has no valid parent.
            log_info("Sentence %d (section %d) of length %d (%d arcs out of %d arcs are correct)", si, sent->section, sent->length, nm, sent->length);

            int sentence_length = sent->length;
            for (int to = 1; to <= sentence_length; to++) {

                if (model[to] != empirical[to]) {

                    update_alpha(mdl, si, model[to], to, sent, -1);

                    update_alpha(mdl, si, empirical[to], to, sent, +1);
                }


            }
        } else {
            log_info("Sentence %d (section %d) of length %d (Perfect parse)", si, sent->section, sent->length);
        }

        size_t nsuccess;
        if (budget_method == RANDOMIZED) {
            if (mdl->M > budget_target) {
                size_t nbefore = mdl->M;
                size_t nasked = nbefore - budget_target;
                nsuccess = delete_n_random_hypothesis(mdl, nasked);

                log_info("%lu vectors deleted (%lu asked). Current hypothesis set size reduced from %lu to %lu", nsuccess, nasked, nbefore, mdl->M);
            }
        }

        mdl->c++;

        free_feature_matrix(corpus, si);

        match += nm;
        total += (sent->length);


        if ((si + 1) % 1000 == 0 && si != 0) {
            log_info("Running training accuracy %lf after %d sentence.", (match * 1.) / total, si + 1);

            unsigned nsv = mdl->M;
            log_info("%u (%f of total %d) support vectors", nsv, (nsv * 1.) / max_sv, max_sv);
        }

        free(model);
        free(empirical);
    }

    unsigned nsv = mdl->M;
    log_info("Running training accuracy %lf", (match * 1.) / total);
    log_info("%u (%f of total %d) support vectors", nsv, (nsv * 1.) / max_sv, max_sv);

    if (verbosity > 0) {

        dump_support_vectors(mdl);



    }

    update_average_alpha(mdl);

    return;
}

void dump_conll_word(Word w, bool true_parent, FILE* ofp) {

    for (int i = 0; i < 10; i++) {

        if (i != 6)
            fprintf(ofp, "%s", (char*) DArray_get(w->conll_piece, i));
        else {
            if (true_parent)
                fprintf(ofp, "%d", w->parent);
            else
                fprintf(ofp, "%d", w->predicted_parent);
        }

        if (i < 9)
            fprintf(ofp, "\t");
    }
    fprintf(ofp, "\n");

}

ParserTestMetric test_KernelPerceptronModel(void *mdl, const CoNLLCorpus corpus, bool use_temp_weight, FILE *gold_ofp, FILE *model_ofp) {
    ParserTestMetric metric = create_ParserTestMetric();

    PerceptronModel linear_mdl;
    KernelPerceptron kernel_mdl;
    if (kernel == KLINEAR) {
        linear_mdl = (PerceptronModel) mdl;
    } else {
        kernel_mdl = (KernelPerceptron) mdl;
    }

    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        FeaturedSentence sent = DArray_get(corpus->sentences, si);

        debug("Test sentence %d (section %d) of length %d", si, sent->section, sent->length);

        if (kernel != KLINEAR) {
                debug("Generating feature matrix for sentence %d", si);
                set_FeatureMatrix(NULL, corpus, si);
        }

        if (kernel == KLINEAR) {
            if (use_temp_weight) {
                debug("\tI will be using a weight vector (raw) of length %ld", linear_mdl->embedding_w_temp->n);
                build_adjacency_matrix(corpus, si, linear_mdl->embedding_w_temp, NULL);
            } else {
                debug("\tI will be using a weight vector (averaged) of length %ld", linear_mdl->embedding_w->n);
                build_adjacency_matrix(corpus, si, linear_mdl->embedding_w, NULL);
            }

        } else {
            debug("Generating adj. matrix for sentence %d", si);
            set_adjacency_matrix_fast(corpus, si, kernel_mdl, true);
        }


        debug("Now parsing sentence %d", si);
        int *model = parse(sent);

        (metric->total_sentence)++;
        debug("Now comparing actual arcs with model generated arcs for sentence %d (Last sentence is %d)", si, sent->length);
        for (int j = 0; j < sent->length; j++) {
            Word w = (Word) DArray_get(sent->words, j);

            w->predicted_parent = model[j + 1];

            // TODO: One file per section idea 
            if (model_ofp != NULL)
                dump_conll_word(w, true, model_ofp);

            if (gold_ofp != NULL)
                dump_conll_word(w, false, gold_ofp);

            if (w->parent == 0 && model[j + 1] == 0)
                (metric->true_root_predicted)++;

            debug("\tTrue parent of word %d (with %s:%s) is %d whereas estimated parent is %d", j + 1, w->postag, w->form, w->parent, model[j + 1]);

            int pmatch_nopunc = 0, ptotal_nopunc = 0, pmatch = 0;
            if (strcmp(w->postag, ",") != 0 && strcmp(w->postag, ":") != 0 && strcmp(w->postag, ".") != 0 && strcmp(w->postag, "``") != 0 && strcmp(w->postag, "''") != 0) {

                if (w->parent == model[j + 1]) {
                    (metric->without_punc->true_prediction)++;
                    pmatch_nopunc++;
                }

                ptotal_nopunc++;

                (metric->without_punc->total_prediction)++;
            }

            if (pmatch_nopunc == ptotal_nopunc && pmatch_nopunc != 0) {
                (metric->complete_sentence_without_punc)++;
            }


            (metric->all->total_prediction)++;

            if (w->parent == model[j + 1]) {
                pmatch++;
                (metric->all->true_prediction)++;
            }

            if (pmatch == sent->length && pmatch != 0)
                (metric->complete_sentence)++;
        }

        if (model_ofp != NULL) {
            fprintf(model_ofp, "\n");
        }

        if (gold_ofp != NULL) {
            fprintf(gold_ofp, "\n");
        }

        free(model);


        debug("Releasing feature matrix for sentence %d", si);

        free_feature_matrix(corpus, si);
    }

    return metric;
}

void mark_best_KernelPerceptronModel(KernelPerceptron kmodel, int numit) {
    kmodel->best_numit = numit;

    if (kmodel->best_alpha_avg != NULL) {
        mkl_free(kmodel->best_alpha_avg);
    }

    if (kmodel->best_kernel_matrix != NULL) {
        mkl_free(kmodel->best_kernel_matrix);
    }

    kmodel->best_alpha_avg = (float*) mkl_64bytes_malloc((kmodel->M) * sizeof (float));
    kmodel->best_kernel_matrix = (float*) mkl_64bytes_malloc((kmodel->N) * (kmodel->M) * sizeof (float));

    for (size_t i = 0; i < (kmodel->M); i++) {

        (kmodel->best_alpha_avg)[i] = (kmodel->alpha_avg)[i];

    }

    size_t mn = (kmodel->N) * (kmodel->M);
#pragma ivdep
#pragma loop_count min(102400)    
    for (size_t i = 0; i < mn; i++) {

        (kmodel->best_kernel_matrix)[i] = (kmodel->kernel_matrix)[i];

    }

    kmodel->best_m = kmodel->M;
}

void dump_KernelPerceptronModel(FILE *fp, KernelPerceptron kp) {

    fprintf(fp, "kernel=%d\n", kp->kernel);
    fprintf(fp, "power=%d\n", kp->power);
    fprintf(fp, "bias=%f\n", kp->bias);

    fprintf(fp, "nsv=%lu\n", kp->best_m);
    fprintf(fp, "edim=%lu\n", kp->N);
    fprintf(fp, "numit=%d\n", kp->best_numit);
    fprintf(fp, "c=%d\n", kp->c);


    for (size_t i = 0; i < kp->best_m; i++) {
        //fprintf(fp, "alpha_avg[%lu]=%f alpha[%lu]=%f beta[%lu]=%f\n", i, (kp->best_alpha_avg)[i],i, (kp->alpha)[i],i, (kp->beta)[i]);
        fprintf(fp, "alpha[%lu]=%f\n", i, (kp->best_alpha_avg)[i]);
    }

    for (size_t i = 0; i < kp->best_m * kp->N; i++) {
        fprintf(fp, "K[%lu]=%f\n", i, (kp->best_kernel_matrix)[i]);
    }
}

PerceptronModel load_PerceptronModel(FILE *fp){
    
    size_t dummy;
    char buffer[1024];
    
    int n = fscanf(fp, "edimension=%lu\n",&dummy);
    check(n == 1,"Embedding dimension can not read");
    
     n = fscanf(fp, "epattern=%s\n",buffer);
    check(n == 1,"Embedding pattern can not read");
    
     n = fscanf(fp, "bestnumit=%lu\n",&dummy);
    check(n == 1,"Best numit can not read");
    
     n = fscanf(fp, "transformation=%s\n",buffer);
    check(n == 1,"Embedding transformation can not read");
    
     n = fscanf(fp, "dimension=%lu\n",&dummy);
    check(n == 1,"Transformed embedding dimension can not read");
    
    PerceptronModel model = create_PerceptronModel(dummy, NULL);
    
    size_t real_idx;
    for (size_t i = 0; i < dummy; i++) {
        n = fscanf(fp, "%lu=%f\n", &real_idx, &((model->embedding_w->data)[i]));

        check(n == 2, "Either index (%lu) or coefficient(%f) is missing", real_idx, (model->embedding_w->data)[i]);
        check(i == real_idx, "Expected index (%lu) does not match with current index(%lu)", i, real_idx);

        (model->embedding_w_avg->data)[i] = (model->embedding_w->data)[i];
        (model->embedding_w_best->data)[i] = (model->embedding_w->data)[i];
        (model->embedding_w_temp->data)[i] = (model->embedding_w->data)[i];
    }
    
    return model;
error:
    return NULL;
}

KernelPerceptron load_KernelPerceptronModel(FILE *fp) {

    enum Kernel type;

    int n = fscanf(fp, "kernel=%d\n", &type);

    debug("Kernel type is %d", type);

    check(n == 1, "No kernel type found in file");

    check(type == KPOLYNOMIAL, "Only POLYNOMIAL is supported. %d is not", type);

    int power;
    float bias;

    n = fscanf(fp, "power=%d\n", &power);
    check(n == 1 && power > 0, "No power found for polynomial model");

    debug("Power is %d", power);
    n = fscanf(fp, "bias=%f\n", &bias);
    check(n == 1, "No bias found for polynomial model");
    debug("Bias is %f", bias);

    KernelPerceptron model = create_PolynomialKernelPerceptron(power, bias);

    n = fscanf(fp, "nsv=%lu\nedim=%lu\nnumit=%d\nc=%d\n", &(model->M), &(model->N), &(model->best_numit), &(model->c));
    check(n == 4, "Num s.v. or embedding dimension or numit or c is missing in model file");

    debug("Number of Support Vectors are %lu", model->M);
    debug("Embedding dimensiob is %lu", model->N);
    debug("Number of Iterations are %d", model->best_numit);
    debug("C is %d", model->c);

    model->alpha_avg = (float*) mkl_64bytes_malloc(model->M * sizeof (float));
    model->alpha = (float*) mkl_64bytes_malloc(model->M * sizeof (float));
    size_t real_idx;
    for (size_t i = 0; i < model->M; i++) {
        n = fscanf(fp, "alpha[%lu]=%f\n", &real_idx, &((model->alpha)[i]));

        check(n == 2, "Either index (%lu) or coefficient(%f) is missing", real_idx, (model->alpha)[i]);
        check(i == real_idx, "Expected index (%lu) does not match with current index(%lu)", i, real_idx);

        (model->alpha_avg)[i] = (model->alpha)[i];
    }

    model->kernel_matrix = (float*) mkl_64bytes_malloc(model->M * model->N * sizeof (float));


    for (size_t i = 0; i < model->M * model->N; i++) {

        n = fscanf(fp, "K[%lu]=%f\n", &real_idx, &((model->kernel_matrix)[i]));

        check(n == 2, "Either index (%lu) or coefficient(%f) is missing", real_idx, (model->kernel_matrix)[i]);
        check(i == real_idx, "Expected index (%lu) does not match with current index(%lu)", i, real_idx);
    }

    return model;
error:
    return NULL;

}