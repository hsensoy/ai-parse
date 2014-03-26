//
//  datastructure.h
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 19/02/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#ifndef Perceptron_GLM_NLP_Tasks_datastructure_h
#define Perceptron_GLM_NLP_Tasks_datastructure_h

struct IntegerIndexedFeatures{
    Hashmap *map;
    uint32_t feature_id;
};

typedef struct IntegerIndexedFeatures* IntegerIndexedFeatures;



#endif
