/* 
 * File:   ai-parse.c
 * Author: husnusensoy
 *
 * Created on March 17, 2014, 6:30 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include "argparse.h"
#include "debug.h"
#include "parseutil.h"
#include "dependency.h"
#include "mkl.h"
#include <string.h>

#define VERSION "v0.9.9.0 (Thomas Jefferson)"


#define DEFAULT_MAX_NUMIT 50
#define DEFAULT_TRAINING_SECTION_STR "2-22"
#define DEFAULT_DEV_SECTION_STR "22"
#define DEFAULT_EMBEDDING_TRANFORMATION LINEAR
#define DEFAULT_KERNEL KLINEAR

static const char *const usage[] = {
    "ai-parse [options] [[--] args]",
    NULL,
};


/**
 * epattern is the embedding pattern.
 * etransform is the embedding vector transformation to be applied.
 * 
 */
const char *epattern = NULL;
enum EmbeddingTranformation etransform = DEFAULT_EMBEDDING_TRANFORMATION;
enum Kernel kernel = DEFAULT_KERNEL;
int num_parallel_mkl_slaves = -1;
const char *modelname = NULL;
enum BudgetMethod budget_method = NONE;
size_t budget_target = 50000;
int polynomial_degree = 4;
float bias = 1.0;
float rbf_lambda = 0.025;
int edimension = 0;

int verbosity = 0;

/*
 * 
 */
int main(int argc, char** argv) {


    int maxnumit = 0;
    int maxrec = -1;

    const char *budget_type_str = NULL;
    const char *stage = NULL;
    const char *training = NULL;
    const char *dev = NULL;
    const char *path = NULL;
    const char * etransform_str = NULL;
    const char *kernel_str = NULL;
    const char *rbf_lambda_str = NULL;

#ifdef NDEBUG
    log_info("ai-parse %s (Release)", VERSION);
#else
    log_info("ai-parse %s (Debug)", VERSION);
#endif

    struct argparse_option options[] = {
        OPT_HELP(),
        //OPT_BOOLEAN('f', "force", &force, "force to do", NULL),
        OPT_INTEGER('v', "verbosity", &verbosity, "Verbosity level. Minimum (Default) 0. Increasing values increase parser verbosity.", NULL),
        OPT_STRING('o', "modelname", &modelname, "Model name", NULL),
        OPT_STRING('p', "path", &path, "CoNLL base directory including sections", NULL),
        OPT_STRING('s', "stage", &stage, "[ optimize | train | parse ]", NULL),
        OPT_INTEGER('n', "maxnumit", &maxnumit, "Maximum number of iterations by perceptron. Default is 50", NULL),
        OPT_STRING('t', "training", &training, "Training sections for optimize and train. Apply sections for parse", NULL),
        OPT_STRING('d', "development", &dev, "Development sections for optimize", NULL),
        OPT_STRING('e', "epattern", &epattern, "Embedding Patterns", NULL),
        OPT_INTEGER('l', "edimension", &edimension, "Embedding dimension", NULL),
        OPT_INTEGER('m', "maxrec", &maxrec, "Maximum number of training instance", NULL),
        OPT_STRING('x', "etransform", &etransform_str, "Embedding Transformation", NULL),
        OPT_STRING('k', "kernel", &kernel_str, "Kernel Type", NULL),
        OPT_INTEGER('a', "bias", &bias, "Polynomial kernel additive term. Default is 1", NULL),
        OPT_INTEGER('c', "concurrency", &num_parallel_mkl_slaves, "Parallel MKL Slaves. Default is 90% of all machine cores", NULL),
        OPT_INTEGER('b', "degree", &polynomial_degree, "Degree of polynomial kernel. Default is 4", NULL),
        OPT_STRING('z', "lambda", &rbf_lambda_str, "Lambda multiplier for RBF Kernel.Default value is 0.025"),
        OPT_STRING('u', "budget_type", &budget_type_str, "Budget control methods. NONE|RANDOM", NULL),
        OPT_INTEGER('g', "budget_size", &budget_target, "Budget Target for budget based perceptron algorithms. Default 50K", NULL),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argc = argparse_parse(&argparse, argc, argv);

    int max_threads = mkl_get_max_threads();
    log_info("There are max %d MKL threads", max_threads);

    if (num_parallel_mkl_slaves == -1) {

        num_parallel_mkl_slaves = (int) (max_threads * 0.9);

        if (num_parallel_mkl_slaves == 0)
            num_parallel_mkl_slaves = 1;

    }

    log_info("Number of MKL Slaves is set to be %d", num_parallel_mkl_slaves);
    mkl_set_num_threads(num_parallel_mkl_slaves);

    if (1 == mkl_get_dynamic())
        log_info("Intel MKL may use less than %i threads for a large problem", num_parallel_mkl_slaves);
    else
        log_info("Intel MKL should use %i threads for a large problem", num_parallel_mkl_slaves);

    check(stage != NULL && (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0 || strcmp(stage, "parse") == 0),
            "Choose one of -s optimize, train, parse");

    check(path != NULL, "Specify a ConLL base directory using -p");

    check(edimension != 0, "Set embedding dimension using -l");

    check(modelname != NULL, "Provide model name using -o");

    if (budget_type_str != NULL) {
        if (strcmp(budget_type_str, "RANDOM") == 0 || strcmp(budget_type_str, "RANDOMIZED") == 0) {
            budget_method = RANDOMIZED;
        } else if (strcmp(budget_type_str, "NONE") == 0) {
            budget_method = NONE;

        } else {
            log_err("Unknown budget control type %s", budget_type_str);
            goto error;
        }

    } else {
        budget_method = NONE;
    }



    if (training == NULL) {
        log_warn("training section string is set to %s", DEFAULT_TRAINING_SECTION_STR);

        training = strdup(DEFAULT_TRAINING_SECTION_STR);
    }

    if (dev == NULL && (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0)) {
        log_info("development section string is set to %s", DEFAULT_DEV_SECTION_STR);

        dev = strdup(DEFAULT_DEV_SECTION_STR);
    }

    check(epattern != NULL, "Embedding pattern is required for -s optimize,train,parse");

    if (etransform_str == NULL) {
        log_info("Embedding transformation is set to be QUADRATIC");

        etransform = DEFAULT_EMBEDDING_TRANFORMATION;
    } else if (strcmp(etransform_str, "LINEAR") == 0) {
        etransform = LINEAR;
    } else if (strcmp(etransform_str, "QUADRATIC") == 0) {
        etransform = QUADRATIC;
    } else if (strcmp(etransform_str, "CUBIC") == 0) {
        etransform = CUBIC;
    } else {
        log_err("Unsupported transformation type for embedding %s", etransform_str);
    }

    if (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0) {

        if (maxnumit <= 0) {
            log_info("maxnumit is set to %d", DEFAULT_MAX_NUMIT);

            maxnumit = DEFAULT_MAX_NUMIT;
        }
    }

    if (kernel_str != NULL) {
        if (strcmp(kernel_str, "POLYNOMIAL") == 0) {

            log_info("Polynomial kernel will be used with bias %f and degree %d", bias, polynomial_degree);

            kernel = KPOLYNOMIAL;
        } else if (strcmp(kernel_str, "GAUSSIAN") == 0 || strcmp(kernel_str, "RBF") == 0) {

            if (rbf_lambda_str != NULL) {
                rbf_lambda = (float) atof(rbf_lambda_str);
            }

            log_info("RBF/GAUSSIAN kernel will be used with lambda %f ", rbf_lambda);

            kernel = KRBF;


        } else {
            log_err("Unsupported kernel type %s. Valid options are LINEAR, POLYNOMIAL, and RBF/GAUSSIAN", kernel_str);
            goto error;
        }
    }

    if (strcmp(stage, "optimize") == 0) {
        void *model = optimize(maxnumit, maxrec, path, training, dev, edimension);

        char* model_filename = (char*) malloc(sizeof (char) * (strlen(modelname) + 7));
        check_mem(model_filename);

        sprintf(model_filename, "%s.model", modelname);

        FILE *fp = fopen(model_filename, "w");

        if (kernel == KLINEAR) {

            PerceptronModel pmodel = (PerceptronModel) model;

            dump_PerceptronModel(fp, edimension, pmodel->embedding_w_best, pmodel->best_numit);

            PerceptronModel_free(pmodel);
        } else if (kernel == KPOLYNOMIAL || kernel == KRBF) {
            KernelPerceptron kpmodel = (KernelPerceptron) model;

            dump_KernelPerceptronModel(fp, kpmodel);
        }

        log_info("Model is dumped into %s file", model_filename);


        fclose(fp);



    } else if (strcmp(stage, "parse") == 0) {
        char* model_filename = (char*) malloc(sizeof (char) * (strlen(modelname) + 7));
        check_mem(model_filename);

        sprintf(model_filename, "%s.model", modelname);
        FILE *fp = fopen(model_filename, "r");

        check(fp != NULL, "%s could not be opened", model_filename);

        void *model;
        if (kernel == KLINEAR)
            model = load_PerceptronModel(fp);
        else
            model = load_KernelPerceptronModel(fp);

        fclose(fp);

        check(model != NULL, "Error in loading model file");

        log_info("Model loaded from %s successfully", model_filename);

        parseall(model, path, training, edimension);
    } else {
        log_info("Waiting for implementation");
    }



    return (EXIT_SUCCESS);
error:

    return (EXIT_FAILURE);

}

