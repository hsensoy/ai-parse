//
//  util.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 03/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_util_h
#define Perceptron_GLM_NLP_Tasks_util_h

#include <stdint.h>
#include <stdio.h>
#include "darray.h"

/**
 * @brief Returns a DArray containing integer sequence [start, end )
 * @param start
 * @param end
 * @return DArray containing all integers [start, end )
 */
DArray* range(int start, int end);
DArray* parse_range(char *rangestr);

void print_range(const char *promt, DArray* range);
char* join_range(DArray *range);

#endif
