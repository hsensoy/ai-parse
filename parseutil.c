
#include "util.h"
#include "corpus.h"
#include "dependency.h"
#include "parseutil.h"
#include <stdbool.h>

#ifdef __GNUC__
#include <signal.h>
#include <sys/types.h>
#endif

static bool keepRunning = true;

void intHandler(int dummy) {
    keepRunning = false;
}

#define STOP_ON_CONVERGE true
#define MAX_IDLE_ITER 3
#define MIN_DELTA 0.001

void kernel_workbench(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension, enum Kernel kernel, int bias, int degree) {
    DArray *train_sections = parse_range(train_sections_str);
    DArray *dev_sections = parse_range(dev_sections_str);
     long match = 0, total = 0;

    signal(SIGINT, intHandler);

    log_info("Development sections to be used in %s: %s", path, join_range(dev_sections));

    CoNLLCorpus dev = create_CoNLLCorpus(path, dev_sections, embedding_dimension, NULL);

    log_info("Training sections to be used in %s: %s", path, join_range(train_sections));

    CoNLLCorpus train = create_CoNLLCorpus(path, train_sections, embedding_dimension, NULL);

    log_info("Reading training corpus");
    read_corpus(train, false);


    read_corpus(dev, false);
    log_info("Dev corpus is loaded..");

    alpha_t *va = NULL;

    int i;
    
    int max_sv = 0;
    for (i = 0; i < DArray_count(train->sentences); i++) {


        FeaturedSentence sent = (FeaturedSentence) DArray_get(train->sentences, i);

        
        set_FeatureMatrix(NULL, train, i);

        set_adjacency_matrix(train, i, &va, KPOLYNOMIAL);
        
        max_sv += (sent->length+1) * sent->length - sent->length;

        int *model = parse(sent);
        
        //printfarch(model, sent->length);
        debug("Parsing sentence %d is done", i);
        int *empirical = get_parents(sent);

        //printfarch(empirical, sent->length);
        int nm = nmatch(model, empirical, sent->length);
        
        debug("Model matches %d arcs out of %d arcs", nm, sent->length);
        if (nm != sent->length) { // root has no valid parent.
            debug("Model matches %d arcs out of %d arcs", nm, sent->length);
            

            for (int to = 1; to <= sent->length; to++) {


                update_alpha(&va, i, model[to], to, sent, -1);

                update_alpha(&va, i, empirical[to], to , sent, +1);


            }
        }
        
        match += nm;
        total += (sent->length);
        
  
        if ((i+1) % 100 == 0 && i != 0) {
            log_info("Running training accuracy %lf after %d sentence.", (match * 1.) / total,i+1);
            
            unsigned nsv  =HASH_COUNT(va);
            log_info("%u (%f of total %d) support vectors", nsv, (nsv*1.)/max_sv,max_sv);
        }


        free(model);
        free(empirical);

        //free_sentence_structures(sent);
    }

    log_info("Running training accuracy %lf", (match * 1.) / total);

    free_CoNLLCorpus(dev, true);
    free_CoNLLCorpus(train, true);
}

PerceptronModel optimize(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension, const char* embedding_pattern, enum EmbeddingTranformation tranformation) {
    DArray *train_sections = parse_range(train_sections_str);
    DArray *dev_sections = parse_range(dev_sections_str);

    signal(SIGINT, intHandler);

    log_info("Development sections to be used in %s: %s", path, join_range(dev_sections));

    CoNLLCorpus dev = create_CoNLLCorpus(path, dev_sections, embedding_dimension, NULL);

    log_info("Training sections to be used in %s: %s", path, join_range(train_sections));

    CoNLLCorpus train = create_CoNLLCorpus(path, train_sections, embedding_dimension, NULL);

    log_info("Reading training corpus");
    read_corpus(train, false);

    log_info("Reading dev corpus");
    read_corpus(dev, false);

    float *numit_dev_avg = (float*) malloc(sizeof (float)* max_numit);
    float *numit_train_avg = (float*) malloc(sizeof (float)*max_numit);

    check(numit_dev_avg != NULL, "Memory allocation failed for numit_dev_avg");
    check(numit_train_avg != NULL, "Memory allocation failed for numit_train_avg");

    PerceptronModel model = PerceptronModel_create(train, NULL);



    int numit;

    int best_iter = -1;
    float best_score = 0.0;

    for (numit = 1; numit <= max_numit && keepRunning; numit++) {
        log_info("BEGIN: Iteration %d", numit);

        train_perceptron_once(model, train, max_rec);

        log_info("END: Iteration %d", numit);

        double dev_acc = test_perceptron_parser(model, dev, true, true);
        //double train_acc = test_perceptron_parser(model, train, true, true);
        double train_acc = 0.0;


        log_info("\n\tnumit=%d accuracy(dev)=%lf accuracy(train)=%lf\n", numit, dev_acc, train_acc);

        numit_dev_avg[numit - 1] = dev_acc;
        numit_train_avg[numit - 1] = train_acc;

        if (best_score < dev_acc) {
            if (best_score + MIN_DELTA > dev_acc)
                log_info("Improvement is less than %f", MIN_DELTA);

            best_score = dev_acc;
            best_iter = numit;

            model->best_numit = numit;
            memcpy(model->embedding_w_best->data, model->embedding_w_temp->data, sizeof (float)*model->embedding_w_best->true_n);
        }

        if (numit - best_iter > MAX_IDLE_ITER && STOP_ON_CONVERGE) {
            log_info("No improvement in last %d iterations", MAX_IDLE_ITER);
            keepRunning = false;
        }
    }

    log_info("Iteration\tAccuracy(dev)\tAccuracy(train)");
    for (int i = 0; i < numit - 1; i++) {
        log_info("%d\t\t%f\t%f%s", i + 1, numit_dev_avg[i], numit_train_avg[i], (i + 1 == best_iter) ? " (*)" : "");
    }

    free_CoNLLCorpus(dev, true);
    free_CoNLLCorpus(train, true);

    return model;

error:
    log_err("Memory allocation error");

    exit(1);

}
