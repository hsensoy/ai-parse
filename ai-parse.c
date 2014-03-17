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
#include <string.h>

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
    const char *stage = NULL;
    const char *training = NULL;
    const char *dev = NULL;
    const char *path = NULL;
    const char *epattern = NULL;
    const char * etransform_str = NULL;
    enum EmbeddingTranformation etransform = QUADRATIC;

    struct argparse_option options[] = {
        OPT_HELP(),
        //OPT_BOOLEAN('f', "force", &force, "force to do", NULL),
        OPT_STRING('p', "path", &path, "CoNLL base directory including sections", NULL),
        OPT_STRING('s', "stage", &stage, "[ optimize | train | parse ]", NULL),
        OPT_INTEGER('n', "maxnumit", &maxnumit, "Maximum number of iterations by perceptron. Default is 30", NULL),
        OPT_STRING('t', "training", &training, "Training sections for optimize and train. Apply sections for parse", NULL),
        OPT_STRING('d', "development", &dev, "Development sections for optimize", NULL),
        OPT_STRING('e', "epattern", &epattern, "Embedding Patterns", NULL),
        OPT_STRING('x', "etransform", &etransform_str, "Embedding Transformation", NULL),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argc = argparse_parse(&argparse, argc, argv);


    check(stage != NULL && (strcmp(stage, "optimize") == 0 || strcmp(stage, "train") == 0 || strcmp(stage, "parse") == 0),
            "Choose one of -s optimize, train, parse");

    check(path != NULL, "Specify a ConLL base directory");

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
        }else if( strcmp(etransform_str, "LINEAR") == 0){
            etransform = LINEAR;
        }else if( strcmp(etransform_str, "QUADRATIC") == 0){
            etransform = QUADRATIC;
        }else{
            log_info("Unsupported transformation type for embedding %s",etransform_str);
            goto error;
        }
                

    }

    if (strcmp(stage, "optimize") == 0) {

        optimize(maxnumit, path, training, dev, epattern, etransform);



    } else {
        log_info("Waiting for implementation");
    }



    return (EXIT_SUCCESS);
error:

    return (EXIT_FAILURE);

}

