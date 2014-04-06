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
#include <string.h>

#define VERSION "v0.9.1"


#define DEFAULT_MAX_NUMIT 30
#define DEFAULT_TRAINING_SECTION_STR "2-22"
#define DEFAULT_DEV_SECTION_STR "22"
#define DEFAULT_EMBEDDING_TRANFORMATION QUADRATIC

static const char *const usage[] = {
    "ai-parse [options] [[--] args]",
    NULL,
};

/*
 * 
 */
int main(int argc, char** argv) {


    int maxnumit = 0;
    int edimension = 0;
    int maxrec = -1;
    const char *stage = NULL;
    const char *training = NULL;
    const char *dev = NULL;
    const char *path = NULL;
    const char *epattern = NULL;
    const char * etransform_str = NULL;
    const char *modelname = NULL;
    enum EmbeddingTranformation etransform = QUADRATIC;
    
    #ifdef NDEBUG
    log_info("ai-parse %s (Release)", VERSION);
    #else
    log_info("ai-parse %s (Debug)", VERSION);
    #endif

    struct argparse_option options[] = {
        OPT_HELP(),
        //OPT_BOOLEAN('f', "force", &force, "force to do", NULL),
        OPT_STRING('o', "modelname", &modelname, "Model name", NULL),
        OPT_STRING('p', "path", &path, "CoNLL base directory including sections", NULL),
        OPT_STRING('s', "stage", &stage, "[ optimize | train | parse ]", NULL),
        OPT_INTEGER('n', "maxnumit", &maxnumit, "Maximum number of iterations by perceptron. Default is 30", NULL),
        OPT_STRING('t', "training", &training, "Training sections for optimize and train. Apply sections for parse", NULL),
        OPT_STRING('d', "development", &dev, "Development sections for optimize", NULL),
        OPT_STRING('e', "epattern", &epattern, "Embedding Patterns", NULL),
        OPT_INTEGER('l', "edimension", &edimension, "Embedding dimension", NULL),
        OPT_INTEGER('m', "maxrec", &maxrec, "Maximum number of training instance", NULL),
        OPT_STRING('x', "etransform", &etransform_str, "Embedding Transformation", NULL),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argc = argparse_parse(&argparse, argc, argv);


    check(stage != NULL && (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0 || strcmp(stage, "parse") == 0),
            "Choose one of -s optimize, train, parse");

    check(path != NULL, "Specify a ConLL base directory using -p");

    check(edimension != 0, "Set embedding dimension using -l");

    check(modelname != NULL, "Provide model name using -o");

    if (maxnumit <= 0) {
        log_warn("maxnumit is set to %d", DEFAULT_MAX_NUMIT);

        maxnumit = DEFAULT_MAX_NUMIT;
    }

    if (training == NULL) {
        log_warn("training section string is set to %s", DEFAULT_TRAINING_SECTION_STR);

        training = strdup(DEFAULT_TRAINING_SECTION_STR);
    }

    if (dev == NULL && (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0)) {
        log_warn("development section string is set to %s", DEFAULT_DEV_SECTION_STR);

        dev = strdup(DEFAULT_DEV_SECTION_STR);
    }

    if (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0) {

        check(epattern != NULL, "Embedding pattern is required for -s optimize,train");

        if (etransform_str == NULL) {
            log_warn("Embedding transformation is set to be QUADRATIC");

            etransform = DEFAULT_EMBEDDING_TRANFORMATION;
        } else if (strcmp(etransform_str, "LINEAR") == 0) {
            etransform = LINEAR;
        } else if (strcmp(etransform_str, "QUADRATIC") == 0) {
            etransform = QUADRATIC;
        } else {
            log_info("Unsupported transformation type for embedding %s", etransform_str);
            goto error;
        }


    }

    if (strcmp(stage, "optimize") == 0) {

        PerceptronModel model = optimize(maxnumit, maxrec, path, training, dev, edimension, epattern, etransform);


        char* model_filename = (char*)malloc(sizeof(char) * (strlen(modelname) + 7));
        check_mem(model_filename);
        
        sprintf(model_filename, "%s.model", modelname);
        
        log_info("Model is dumped into %s file",model_filename);
        
        FILE *fp = fopen(model_filename, "w");

        dump_model(fp, edimension, epattern, model->embedding_w_best,model->best_numit,etransform);

        fclose(fp);

        PerceptronModel_free(model);



    } else {
        log_info("Waiting for implementation");
    }



    return (EXIT_SUCCESS);
error:

    return (EXIT_FAILURE);

}

