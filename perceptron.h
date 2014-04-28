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

double test_KernelPerceptronModel(KernelPerceptron mdl, const CoNLLCorpus corpus, bool exclude_punct);

 void mark_best_KernelPerceptronModel(KernelPerceptron kmodel, int numit);



#endif	/* PERCEPTRON_H */

