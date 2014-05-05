#include "perceptron.h"
#include "debug.h"
#include "uthash.h"
#include "corpus.h"
#include "dependency.h"
#include "memman.h"

KernelPerceptron create_PolynomialKernelPerceptron(int power, float bias) {

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

void update_alpha(KernelPerceptron kp, uint32_t sidx, uint16_t from, uint16_t to, struct FeaturedSentence* sent, float inc) {

    unsigned keylen;
    alpha_t *a, *dummy = NULL;
    alpha_key_t *a_key = NULL;

    keylen = offsetof(alpha_t, to) /* offset of last key field */
            + sizeof (uint16_t) /* size of last key field */
            - offsetof(alpha_t, sentence_idx); /* offset of first key field */

    a_key = (alpha_key_t*) malloc(sizeof (alpha_key_t));
    memset(a_key, 0, sizeof (alpha_key_t));

    a_key->from = from;
    a_key->to = to;
    a_key->sentence_idx = sidx;



    HASH_FIND(hh, kp->arch_to_index_map, &a_key->sentence_idx, keylen, a);
    if (a != NULL) {
        (kp->alpha)[a->idx] += inc;

        (kp->beta)[a->idx] += inc * kp->c;

    } else {
        vector v = embedding_feature(sent, from, to, NULL);
        size_t n = (kp->M);



        if (n == 0) {
            kp->N = v->true_n;
            kp->kernel_matrix = (float*) mkl_64bytes_malloc((n + 1) * v->true_n * sizeof (float));

            for (size_t i = 0; i < v->true_n; i++)
                (kp->kernel_matrix)[n * v->true_n + i] = v->data[i];

            kp->alpha = (float*) mkl_64bytes_malloc((n + 1) * sizeof (float));
            kp->beta = (float*) mkl_64bytes_malloc((n + 1) * sizeof (float));
            (kp->alpha)[n] = inc;
            (kp->beta)[n] = inc * kp->c;
        } else {
            check(kp->N == v->true_n, "%lu dimensional embedding does not confirm with the previous embedding size (%lu)", v->true_n, kp->N);

            kp->kernel_matrix = (float*) mkl_64bytes_realloc(kp->kernel_matrix, (n + 1) * v->true_n * sizeof (float));

            for (size_t i = 0; i < v->true_n; i++)
                (kp->kernel_matrix)[n * v->true_n + i] = v->data[i];

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

void update_average_alpha(KernelPerceptron kp) {

    if (kp->alpha_avg != NULL) {
        mkl_free(kp->alpha_avg);
    }

    kp->alpha_avg = (float*) mkl_64bytes_malloc((kp->M) * sizeof (float));

    for (size_t i = 0; i < (kp->M); i++) {

        (kp->alpha_avg)[i] = (kp->alpha)[i] - (kp->beta)[i] / (kp->c);

    }

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

    update_average_alpha(mdl);

    return;
}

double test_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, bool exclude_punct, FILE *ofp) {
    int match = 0, total = 0;
    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        FeaturedSentence sent = DArray_get(corpus->sentences, si);

        log_info("Test sentence %d (section %d) of length %d", si, sent->section, sent->length);

        debug("Generating feature matrix for sentence %d", si);
        set_FeatureMatrix(NULL, corpus, si);

        debug("Generating adj. matrix for sentence %d", si);
        set_adjacency_matrix_fast(corpus, si, mdl, true);

        debug("Now parsing sentence %d", si);
        int *model = parse(sent);


        debug("Now comparing actual arcs with model generated arcs for sentence %d (Last sentence is %d)", si, sent->length);
        for (int j = 0; j < sent->length; j++) {
            Word w = (Word) DArray_get(sent->words, j);
            int p = w->parent;
            char *postag = w->postag;

            if (ofp != NULL) {
                fprintf(ofp, "%d\t%s\t%s\t%d\t%d\t%u\n", j + 1, w->form, postag, p, model[j + 1], sent->section);
            }

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

        if (ofp != NULL) {
            fprintf(ofp, "\n");
        }

        free(model);


        debug("Releasing feature matrix for sentence %d", si);

        free_feature_matrix(corpus, si);


    }

    return (match * 1.) / total;
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

KernelPerceptron load_KernelPerceptronModel(FILE *fp) {

    enum Kernel type;

    int n = fscanf(fp, "kernel=%d\n", &type);
    
    debug("Kernel type is %d",type);

    check(n == 1, "No kernel type found in file");

    check(type == KPOLYNOMIAL, "Only POLYNOMIAL is supported. %d is not", type);

    int power;
    float bias;

    n = fscanf(fp, "power=%d\n", &power);
    check(n == 1 && power > 0, "No power found for polynomial model");
    
    debug("Power is %d",power);
    n = fscanf(fp, "bias=%f\n", &bias);
    check(n == 1, "No bias found for polynomial model");
    debug("Bias is %f",bias);

    KernelPerceptron model = create_PolynomialKernelPerceptron(power, bias);

    n = fscanf(fp, "nsv=%lu\nedim=%lu\nnumit=%d\nc=%d\n", &(model->M), &(model->N), &(model->best_numit), &(model->c));
    check(n == 4, "Num s.v. or embedding dimension or numit or c is missing in model file");

    debug("Number of Support Vectors are %lu",model->M);
    debug("Embedding dimensiob is %lu",model->N);
    debug("Number of Iterations are %d",model->best_numit);
    debug("C is %d",model->c);
    
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