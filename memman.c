//
//  memman.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "memman.h"
#include "debug.h"

size_t aligned_size(size_t n){
    return ((n/32) + 1) * 32;
}

float* alloc_aligned(size_t n){
    float *buffer;
	
	
	/*
		int rc = posix_memalign((void**)&buffer, 32, sizeof(float) * n);
    
		check(rc == 0, "Buffer allocation error");
    
		for (int i = 0 ;i < n;i++)
			buffer[i] = 0.0;
	*/
		 
	buffer = calloc(n, sizeof(float));
	check_mem(buffer);
    
    return buffer;
    
error:
    exit(1);
}



