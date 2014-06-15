
#include "util.h"
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

/**
 * ai-parse.c file for actual storage allocation for those two variables
 */
extern const char *epattern;
extern enum EmbeddingTranformation etransform;
extern enum Kernel kernel;
extern const char *modelname;
extern int polynomial_degree;
extern float bias;
extern float rbf_lambda;

void* optimize(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension) {
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

    PerceptronModel model = NULL;
    KernelPerceptron kmodel = NULL;
    if (kernel == KLINEAR)
        model = PerceptronModel_create(train, NULL);
    else if (kernel == KPOLYNOMIAL)
        kmodel = create_PolynomialKernelPerceptron(polynomial_degree, bias);
    else
        kmodel = create_RBFKernelPerceptron(rbf_lambda);


    int numit;

    int best_iter = -1;
    float best_score = 0.0;

    for (numit = 1; numit <= max_numit && keepRunning; numit++) {
        log_info("BEGIN-TRAIN: Iteration %d", numit);

        if (kernel == KLINEAR)
            train_perceptron_once(model, train, max_rec);
        else
            train_once_KernelPerceptronModel(kmodel, train, max_rec);


        log_info("END-TRAIN: Iteration %d", numit);

        ParserTestMetric dev_metric;
        log_info("BEGIN-TEST: Iteration %d", numit);
        if (kernel == KLINEAR)
            dev_metric = test_perceptron_parser(model, dev, true, true);
        else
            dev_metric = test_KernelPerceptronModel(kmodel, dev, true, NULL, NULL);
        log_info("END-TEST: Iteration %d", numit);

        log_info("\nnumit=%d", numit);

        printParserTestMetric(dev_metric);

        double dev_acc = (dev_metric->without_punc->true_prediction * 1.) / dev_metric->without_punc->total_prediction;
        numit_dev_avg[numit - 1] = dev_acc;
        numit_train_avg[numit - 1] = 0.0;

        freeParserTestMetric(dev_metric);

        if (best_score < dev_acc) {
            if (best_score + MIN_DELTA > dev_acc)
                log_warn("Improvement is less than %f", MIN_DELTA);

            best_score = dev_acc;
            best_iter = numit;

            if (kernel == KLINEAR)
                mark_best_PerceptronModel(model, numit);
            else
                mark_best_KernelPerceptronModel(kmodel, numit);
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

    //free_CoNLLCorpus(dev, true);
    //free_CoNLLCorpus(train, true);

    if (kernel == KLINEAR)
        return (void*) model;
    else
        return (void*) kmodel;

error:
    log_err("Memory allocation error");

    exit(1);

}

void parseall(const KernelPerceptron model, const char* path, const char* test_sections_str, int embedding_dimension) {
    DArray *test_sections = parse_range(test_sections_str);

    signal(SIGINT, intHandler);

    log_info("Test sections to be used in %s: %s", path, join_range(test_sections));

    CoNLLCorpus test = create_CoNLLCorpus(path, test_sections, embedding_dimension, NULL);

    log_info("Reading test corpus");
    read_corpus(test, false);

    int numit;

    int best_iter = -1;
    float best_score = 0.0;

    char* output_filename = (char*) malloc(sizeof (char) * (strlen(modelname) + 13));
    check_mem(output_filename);

    sprintf(output_filename, "%s.conll.gold", modelname);
    FILE *gold_fp = fopen(output_filename, "w");

    sprintf(output_filename, "%s.conll.model", modelname);
    FILE *model_fp = fopen(output_filename, "w");
    ParserTestMetric test_metric = test_KernelPerceptronModel(model, test, true, gold_fp, model_fp);
    fclose(gold_fp);
    fclose(model_fp);


    printParserTestMetric(test_metric);
    freeParserTestMetric(test_metric);

    free(output_filename);

    return;
error:
    log_err("Memory allocation error");

    exit(1);


}
