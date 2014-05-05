//
//  memman.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "memman.h"
#include "debug.h"
#include "mkl.h"

size_t aligned_size(size_t n) {
    return ((n / 32) + 1) * 32;
}

float* alloc_aligned(size_t n) {
    float *buffer;

    buffer = calloc(n, sizeof (float));
    check_mem(buffer);

    return buffer;

error:
    exit(1);
}

void* mkl_64bytes_malloc(size_t bytes) {
    void *buffer = mkl_malloc(bytes, 64);

    check(buffer != NULL, "Memory allocation error...");

    return buffer;
error:
    mkl_free(buffer);
    exit(1);
}

void* mkl_64bytes_realloc(void* ptr, size_t newbytes) {
    void *buffer;
    if (ptr == NULL){
        buffer = mkl_64bytes_malloc(newbytes);
    }else{
        buffer = mkl_realloc(ptr, newbytes);

        check(buffer != NULL, "Memory allocation error...");
    }

    return buffer;
error:
    mkl_free(buffer);
    exit(1);
}





