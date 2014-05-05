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
#include <stdint.h>


#ifdef	__cplusplus
extern "C" {
#endif




#ifdef	__cplusplus
}
#endif




KernelPerceptron create_PolynomialKernelPerceptron(int power, float bias);

void update_alpha(KernelPerceptron kp, uint32_t sidx, uint16_t from, uint16_t to,  FeaturedSentence sent, float inc);

void update_average_alpha(KernelPerceptron kp);

void train_once_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, int max_rec);

/**
 * 
 * @param mdl KernelPerceptron model
 * @param corpus Corpus to be parsed
 * @param exclude_punct Whether to exclude punctuation head estimation in evaluation.
 * @param ofp Dump the parsed sentences into a file if ofp != NULL
 * @return accuracy 
 */
double test_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, bool exclude_punct,FILE *ofp);

void mark_best_KernelPerceptronModel(KernelPerceptron kmodel, int numit);
 
void dump_KernelPerceptronModel(FILE *fp, KernelPerceptron kp);
KernelPerceptron load_KernelPerceptronModel(FILE *fp);



#endif	/* PERCEPTRON_H */

