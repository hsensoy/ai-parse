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

size_t aligned_size(size_t n);

/**
 * @brief Allocates an 32 byte (8 4-byte floats) aligned memory block using posix_memalign
 * 			BUG: posix_memalign causes valgrind to catch memory leaks. This should be a problem with Mac OS compatibility problem.
 * @param n A multiple of 32
 * @return Address of allocated memory block
 */
float* alloc_aligned(size_t n);


void* mkl_64bytes_malloc(size_t bytes);
void* mkl_64bytes_realloc(void* ptr, size_t newbytes);



#endif
