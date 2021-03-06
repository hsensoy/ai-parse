/* 
 * File:   perceptron.h
 * Author: husnusensoy
 *
 * Created on April 27, 2014, 2:51 PM
 */

#ifndef PERCEPTRON_H
#define	PERCEPTRON_H
#include "mkl.h"
#include "datastructure.h"
#include "debug.h"
#include "corpus.h"
#include "dependency.h"
#include <stdint.h>


#ifdef	__cplusplus
extern "C" {
#endif




#ifdef	__cplusplus
}
#endif
 
/**
 * 
 * NONE: No budgeting at all
 * RANDOMIZED: Choose one randomly out of hypothesis vector with alpha value equal to 1.
 */
enum BudgetMethod{
    NONE,
    RANDOMIZED
};




KernelPerceptron create_PolynomialKernelPerceptron(int power, float bias);
KernelPerceptron create_RBFKernelPerceptron(float lambda) ;

void update_alpha(KernelPerceptron kp, uint32_t sidx, uint16_t from, uint16_t to,  FeaturedSentence sent, float inc);

void update_average_alpha(KernelPerceptron kp);

void train_once_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, int max_rec);

/**
 * 
 * @param mdl KernelPerceptron model
 * @param corpus Corpus to be parsed
 * @param use_temp_weight Whether to use averaged weights or the raw (use_temp_weight=true) ones 
 * @param gold_ofp Dump the parsed sentences with gold parent estimations into a file if gold_ofp != NULL
 * @param model_ofp Dump the parsed sentences with model parent estimations into a file if model_ofp != NULL
 * @return ParserTestMetric struct 
 */
ParserTestMetric test_KernelPerceptronModel(void *mdl, const CoNLLCorpus corpus, bool use_temp_weight, FILE *gold_ofp, FILE *model_ofp);

void mark_best_KernelPerceptronModel(KernelPerceptron kmodel, int numit);
 
void dump_KernelPerceptronModel(FILE *fp, KernelPerceptron kp);
KernelPerceptron load_KernelPerceptronModel(FILE *fp);


PerceptronModel load_PerceptronModel(FILE *fp);




#endif	/* PERCEPTRON_H */

