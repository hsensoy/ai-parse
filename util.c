//
//  util.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 03/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "util.h"
#include "darray.h"
#include "stringalgo.h"

#include <assert.h>
#include <stdlib.h>


DArray* range(int start, int end){
    DArray *result = DArray_create(sizeof(int), end - start);
    check_mem(result);
    
    for (int v=start; v < end; v++) {
        int* i = (int*)malloc(sizeof(int));
        check_mem(i);
        
        *i = v;
        
        DArray_push(result, i);
    }
    
    return result;
error:
    exit(1);
}

DArray* parse_range(char *rangestr){
    DArray *result = DArray_create(sizeof(int), 1);
    check_mem(result);
    
    if (strchr(rangestr, '-') != NULL){
        DArray *tokens = split(rangestr, "-");
        
        check(DArray_count(tokens) == 2, "Invalid range string %s %d", rangestr, DArray_count(tokens) == 2);
        
        int start = atoi((char*)DArray_get(tokens, 0));
        int end = atoi((char*)DArray_get(tokens, 1));
        
        DArray_clear_destroy(tokens);
        
        return range(start, end);
        
    }else if (strchr(rangestr, ',') != NULL){
        DArray *tokens = split(rangestr, ",");
        
        for(int i = 0;i < DArray_count(tokens);i++){
            int *iptr = (int*)malloc(sizeof(int));
            check_mem(iptr);
            
            *iptr = atoi((char*)DArray_get(tokens, i));
            
            DArray_push(result, iptr);
        }
        
        DArray_clear_destroy(tokens);
    }else{
        int *iptr = (int*)malloc(sizeof(int));
        check_mem(iptr);
        
        *iptr = atoi(rangestr);
        
        DArray_push(result, iptr);
    }
    
    
    return result;
error:
    exit(1);
}

char* join_range(DArray *range){
    char* buffer = (char*)malloc(sizeof(char) * 1024);
    check_mem(buffer);
    
    buffer[0] = '\0';
    
    for (int i = 0 ; i < DArray_count(range); i++) {
        
        if (i == 0 ){
            sprintf(buffer, "%d", *((int*)DArray_get(range, i)));
        }else{
            sprintf(buffer, "%s, %d", buffer, *((int*)DArray_get(range, i)));
        }
    }
    
    return buffer;
error:
    exit(1);
}



 
 
 
 



