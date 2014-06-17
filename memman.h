//
//  memman.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_memman_h
#define Perceptron_GLM_NLP_Tasks_memman_h

#include <stdlib.h>

void* mkl_64bytes_malloc(size_t bytes);
void* mkl_64bytes_realloc(void* ptr, size_t newbytes);



#endif
