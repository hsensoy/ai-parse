/* 
 * File:   parseutil.h
 * Author: husnusensoy
 *
 * Created on March 17, 2014, 7:43 PM
 */

#ifndef PARSEUTIL_H
#define	PARSEUTIL_H

#include "corpus.h"

#ifdef	__cplusplus
extern "C" {
#endif




#ifdef	__cplusplus
}
#endif

/**
 * 
 * @param max_numit Maximum number of iterations to go
 * @param path ConLL base directory path
 * @param train_sections_str Training sections
 * @param dev_sections_str Development sections
 * @param embedding_pattern Embedding pattern to be used
 * @param tranformation Transformation to be applied on embedding vector.
 */
void optimize(int max_numit, const char* path, const char* train_sections_str, const char* dev_sections_str, const char* embedding_pattern, enum EmbeddingTranformation tranformation) ;
 

#endif	/* PARSEUTIL_H */

