/* 
 * File:   parseutil.h
 * Author: husnusensoy
 *
 * Created on March 17, 2014, 7:43 PM
 */

#ifndef PARSEUTIL_H
#define	PARSEUTIL_H
#include "perceptron.h"
#include "dependency.h"
#include "corpus.h"

#ifdef	__cplusplus
extern "C" {
#endif




#ifdef	__cplusplus
}
#endif

/**
 * @param max_numit Maximum number of iterations to go
 * @param max_rec Maximum number of records to be used for training
 * @param path ConLL base directory path
 * @param train_sections_str Training sections
 * @param dev_sections_str Development sections
 * @param embedding_dimension Embedding dimension per word
 * 
 * @return Model trained
 */
void* optimize(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension);

void kernel_workbench(int max_numit, int max_rec, const char* path, const char* train_sections_str, const char* dev_sections_str, int embedding_dimension,  enum Kernel kernel, int bias, int degree);

#endif	/* PARSEUTIL_H */

