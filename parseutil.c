
#include "util.h"
#include "corpus.h"
#include "dependency.h"
#include <stdbool.h>

#ifdef __GNUC__
#include <signal.h>
#endif

static bool keepRunning = true;

void intHandler(int dummy) {
    keepRunning = false;
}

#define STOP_ON_CONVERGE true
#define MAX_IDLE_ITER 3
#define MIN_DELTA 0.001

PerceptronModel optimize(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension, const char* embedding_pattern, enum EmbeddingTranformation tranformation) {
    DArray *train_sections = parse_range(train_sections_str);
    DArray *dev_sections = parse_range(dev_sections_str);

    signal(SIGINT, intHandler);

    log_info("Development sections to be used in %s: %s", path, join_range(dev_sections));

    CoNLLCorpus dev = create_CoNLLCorpus(path, dev_sections, embedding_dimension, embedding_pattern, tranformation, NULL);

    log_info("Training sections to be used in %s: %s", path, join_range(train_sections));

    CoNLLCorpus train = create_CoNLLCorpus(path, train_sections, embedding_dimension, embedding_pattern, tranformation, NULL);

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
            memcpy(model->embedding_w_best->data, model->embedding_w_temp->data,sizeof(float)*model->embedding_w_best->true_n);
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
    
    //free_CoNLLCorpus(dev,true);
    //free_CoNLLCorpus(train,true);

    return model;

error:
    log_err("Memory allocation error");

    exit(1);

}
