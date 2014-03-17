//
//  corpus.c
//  Perceptron GLM NLP Tasks
//
//  Created by husnu sensoy on 13/01/14.
//  Copyright (c) 2014 husnu sensoy. All rights reserved.
//

#include "corpus.h"
#include <dirent.h>
#include "stringalgo.h"

#include "debug.h"
#include "dependency.h"

Word Root = NULL;

Word ROOT() {
    if (Root == NULL) {
        Root = (Word) malloc(sizeof (struct Word));
        check_mem(Root);

        Root->id = 0;
        Root->form = strdup("ROOT");
        Root->postag = strdup("ROOT");
        Root->parent = -1;


        Root->embedding = NULL;

    }


    return Root;


error:

    exit(1);
}

CoNLLCorpus create_CoNLLCorpus(const char* base_dir, DArray *sections, const char *embedding_pattern, enum EmbeddingTranformation transform, DArray* discrete_patterns) {
    CoNLLCorpus corpus = (CoNLLCorpus) malloc(sizeof (struct CoNLLCorpus));

    check_mem(corpus);

    corpus->base_dir = base_dir;
    corpus->sections = sections;

    corpus->sentences = DArray_create(sizeof (FeaturedSentence), 2000);
    check_mem(corpus->sentences);

    if (embedding_pattern) {
        corpus->embedding_pattern_parts = split(embedding_pattern, "_");
        corpus->hasembeddings = true;
    } else {
        corpus->embedding_pattern_parts = NULL;
        corpus->hasembeddings = false;
    }

    if (discrete_patterns) {
        corpus->disrete_patterns_parts = DArray_create(sizeof (DArray*), DArray_count(discrete_patterns));
        check_mem(corpus->disrete_patterns_parts);

        for (int i = 0; i < DArray_count(discrete_patterns); i++)
            DArray_push(DArray_get(corpus->disrete_patterns_parts, i), split(((char*) DArray_get(discrete_patterns, i)), "_"));


    } else
        corpus->disrete_patterns_parts = NULL;

    corpus->Root = ROOT();
    corpus->embedding_length = -1;
    corpus->transformed_embedding_length = -1;
    corpus->feature_matrix_singleton = NULL;
    corpus->embedding_transform = transform;


    return corpus;
error:
    exit(1);
}

/**
 * 
 * @param has_discrete_features 
 * @param embedding_length      Length of the embedding vector to be used.
 * @return     Creates a feature vector as union of discrete and continous vectors.
 */
FeatureVector FeatureVector_create(bool has_discrete_features, uint32_t embedding_length) {
    FeatureVector fv = (FeatureVector) malloc(sizeof (struct FeatureVector));
    
    check(fv != NULL, "Error in allocating a FeatureVector");

    if (has_discrete_features) {
        fv->discrete_v = DArray_create(sizeof (uint32_t), 18);
        check_mem(fv->discrete_v);
    } else {
        fv->discrete_v = NULL;
    }

    if (embedding_length > 0) {
        fv->continous_v = vector_create(embedding_length);
    } else {
        fv->continous_v = NULL;
    }

    return fv;
error:
    exit(1);
}

void free_FeatureVector(FeatureVector v){
    
    DArray_destroy(v->discrete_v);
    
    vector_free(v->continous_v);
    
    free(v);
    
}

FeatureMatrix FeatureMatrix_create(int sent_length, uint32_t embedding_length, bool has_discrete_features) {
    FeatureMatrix matrix = (FeatureMatrix) malloc(sizeof (struct FeatureMatrix));
    
    check(matrix != NULL, "Error in allocating matrix FeatureMatrix");

    matrix->matrix_data = (FeatureVector**) malloc(sizeof (FeatureVector*) * (sent_length + 1));
    check(matrix->matrix_data != NULL, "Error in allocating 2-dimensional FeatureVector");

    matrix->size = sent_length + 1;
    matrix->embedding_length = embedding_length;
    matrix->has_discrete_features = has_discrete_features;


    log_info("Allocating a %d x %d FeatureMatrix", matrix->size, matrix->size);
    for (int i = 0; i < matrix->size; i++) {
        (matrix->matrix_data)[i] = (FeatureVector*) malloc(sizeof (FeatureVector) * matrix->size);
        
        check((matrix->matrix_data)[i] != NULL, "Error in allocating FeatureVector row");

        for (int j = 0; j < matrix->size; j++) {
            if (i == j || j == 0)
                (matrix->matrix_data)[i][j] = NULL;
            else{
                (matrix->matrix_data)[i][j] = FeatureVector_create(has_discrete_features, embedding_length);
            }
        }
    }
    
    return matrix;

error:
    exit(1);
}

void free_featureMatrix(FeatureMatrix matrix){
   for (int i = 0; i < matrix->size; i++) {

        for (int j = 0; j < matrix->size; j++) {
            if (i == j || j == 0)
                (matrix->matrix_data)[i][j] = NULL;
            else{
                free_FeatureVector((matrix->matrix_data)[i][j]);
            }
        }
        
        free((matrix->matrix_data)[i]);
    }
   
   free(matrix->matrix_data);
   free(matrix);
}

void build_embedding_feature(FeaturedSentence sent, int from, int to, DArray *patterns) {
    vector bigvector = NULL;

    check(from != to && from <= sent->length && from >= 0 && to >= 1 && to <= sent->length, "Arc between suspicious words %d to %d for sentence length %d", from, to, sent->length);

    debug("Number of embedding patterns is %d", DArray_count(patterns));
    for (int pi = 0; pi < DArray_count(patterns); pi++) {
        char *pattern = (char*) DArray_get(patterns, pi);
        char node;
        char subnode;
        int offset;

        if (strcmp(pattern, "tl") == 0) { //thresholded-length
            node = 'l';
            subnode = 't';
        } else if (strcmp(pattern, "nl") == 0) { // normalized-length
            node = 'l';
            subnode = 'n';
        } else if (strcmp(pattern, "l") == 0) { // raw length
            node = 'l';
            subnode = 'r';
        } else {

            int n = sscanf(pattern, "%c%dv", &node, &offset);

            check(n == 2, "Expected pattern format is [p|c]<offset>v where as got %s", pattern);
            check(node == 'p' || node == 'c', "Unknown node name %c expected p or c", node);
        }

        if (node == 'p') {

            if (from == 0)
                bigvector = vconcat(bigvector, Root->embedding);
            else if (from + offset >= 1 && from + offset <= sent->length) {
                bigvector = vconcat(bigvector, ((Word) DArray_get(sent->words, from + offset - 1))->embedding);
            } else {
                bigvector = vconcat(bigvector, Root->embedding);
            }
        } else if (node == 'c') {

            if (to + offset >= 1 && to + offset <= sent->length) {
                bigvector = vconcat(bigvector, ((Word) DArray_get(sent->words, to + offset - 1))->embedding);
            } else {
                bigvector = vconcat(bigvector, Root->embedding);
            }
        } else if (node == 'l') {

            if (subnode == 't') {
                vector length_v = vector_create(6);
                int threshold_arr[] = {2, 5, 10, 20, 30, 40};

                for (int i = 0; i < 6; i++)
                    if (abs(from - to) > threshold_arr[i])
                        length_v->data[i] = 1;
                    else
                        length_v->data[i] = 0;

                bigvector = vconcat(bigvector, length_v);
            } else if (subnode == 'r') {
                vector length_v = vector_create(1);

                length_v->data[0] = abs(from - to);

                bigvector = vconcat(bigvector, length_v);

            } else if (subnode == 'n') {

                vector length_v = vector_create(1);

                // TODO: Are you an idiot ?
                length_v->data[0] = abs(from - to) / 250.;

                bigvector = vconcat(bigvector, length_v);

            } else {
                log_err("Unknown option with length %c. Valid values are (r)aw-length/(n)ormalized length/(t)hresholded length", subnode);
                exit(EXIT_FAILURE);
            }

        } else {
            log_err("Unknown pattern %s for embedding feature", pattern);
            exit(EXIT_FAILURE);
        }
    }


    vquadratic(((sent->feature_matrix_ref->matrix_data)[from][to])->continous_v, bigvector, 1);
    
    return;
error:
    exit(1);
}

void set_FeatureMatrix(Hashmap* featuremap, CoNLLCorpus corpus, int sentence_idx) {

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    if (corpus->feature_matrix_singleton->size < length){
        log_info( "Singleton Matrix is too small (%d) for a sentence length of (%d). Growing...", corpus->feature_matrix_singleton->size, length);
    
        free_featureMatrix(corpus->feature_matrix_singleton);
        
        // TODO: (length + 1) seems to be a bug.
        corpus->feature_matrix_singleton = FeatureMatrix_create(length+1, corpus->transformed_embedding_length,false);
    }
    
    sentence->feature_matrix_ref = corpus->feature_matrix_singleton;

    

    //check(corpus->feature_matrix_singleton->size >= length, "Singleton Matrix is too small (%d) for a sentence length of (%d). Fix and recompile the code", corpus->feature_matrix_singleton->size, length);

    for (int _from = 0; _from <= length; _from++)
        for (int _to = 1; _to <= length; _to++) {
            if (_to != _from) {
                if (corpus->disrete_patterns_parts)
                    (sentence->feature_matrix_ref->matrix_data)[_from][_to]->discrete_v = NULL;

                if (corpus->hasembeddings) {
                    build_embedding_feature(sentence, _from, _to, corpus->embedding_pattern_parts);


                    if ((sentence_idx + 1) % 1000 == 0)
                        debug("Embedding vector length is %ld", (sentence->feature_matrix_ref->matrix_data)[_from][_to]->continous_v->true_n);

                    //if ((sentence->feature_matrix)[_from][_to]->continous_v->true_n > 50)
                    //	log_info("%ld",(sentence->feature_matrix)[_from][_to]->continous_v->true_n);
                }

            }
        }
    
    return;
}

void free_feature_matrix(CoNLLCorpus corpus, int sentence_idx) {
    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    //int length = sentence->length;

    sentence->feature_matrix_ref = NULL;
}

float** square_adjacency_matrix(int n, float init_value) {

    float** matrix = (float**) malloc(sizeof (float*) * n);
    check_mem(matrix);
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*) malloc(sizeof (float) * n);

        for (int j = 0; j < n; j++) {
            matrix[i][j] = init_value;
        }

        check_mem(matrix[i]);
    }

    return matrix;
error:
    log_err("adjacency_matrix allocation error");
    exit(1);
}

void build_adjacency_matrix(CoNLLCorpus corpus, int sentence_idx, vector embeddings_w, vector discrete_w) {

    FeaturedSentence sentence = (FeaturedSentence) DArray_get(corpus->sentences, sentence_idx);
    int length = sentence->length;

    if (sentence->adjacency_matrix == NULL)
        sentence->adjacency_matrix = square_adjacency_matrix(length + 1, NEGATIVE_INFINITY);


    //sentence->feature_matrix = FeatureMatrix_create(length, corpus->hasembeddings, corpus->disrete_patterns_parts != NULL);

    for (int _from = 0; _from <= length; _from++)
        for (int _to = 1; _to <= length; _to++) {
            if (_to != _from) {
                (sentence->adjacency_matrix)[_from][_to] = 0.0;
                if (corpus->disrete_patterns_parts)
                    (sentence->adjacency_matrix)[_from][_to] = -1; // TODO: Complete discrete dot product.

                if (corpus->hasembeddings) {

                    debug("%d->%d\n", _from, _to);
                    vector embedding = (sentence->feature_matrix_ref->matrix_data)[_from][_to]->continous_v;

                    if (embedding == NULL) {
                        log_err("NULL continous vector");
                        exit(EXIT_FAILURE);
                    }

                    //vprint(embedding);
                    (sentence->adjacency_matrix)[_from][_to] += vdot(embeddings_w, embedding);

                }

            }
        }
}

Word parse_word(char* line, bool hasembeedings) {
    DArray* tokens = split(line, "\t");

    Word w = (Word) malloc(sizeof (struct Word));
    check_mem(w);

    w->id = atoi((char*) DArray_get(tokens, 0));
    free((char*) DArray_get(tokens, 0));

    w->form = (char*) DArray_get(tokens, 1);
    w->postag = (char*) DArray_get(tokens, 3);

    w->parent = atoi((char*) DArray_get(tokens, 6));
    free((char*) DArray_get(tokens, 6));


    // TODO: No relation on arcs
    w->relation = NULL; //strdup(tokens[7]);


    if (hasembeedings) {
        check(DArray_count(tokens) >= 11, "CoNLL files in corpus with embeddings should contain at least 11 fields. 11. field being the embedding field. Found a line with only %d fields", DArray_count(tokens));

        w->embedding = parse_vector((char*) DArray_get(tokens, 10));
        free((char*) DArray_get(tokens, 10));
    } else
        w->embedding = NULL;

    free((char*) DArray_get(tokens, 2));
    free((char*) DArray_get(tokens, 4));
    free((char*) DArray_get(tokens, 5));
    free((char*) DArray_get(tokens, 7));
    free((char*) DArray_get(tokens, 8));
    free((char*) DArray_get(tokens, 9));

    // TODO: This may cause problem ?!?
    DArray_destroy(tokens);

    return w;

error:
    exit(1);
}

void Word_free(Word w) {
    vector_free(w->embedding);
    free(w);
}

void add_word(FeaturedSentence sentence, Word word) {

    DArray_push(sentence->words, word);

    sentence->length++;
}

FeaturedSentence FeatureSentence_create() {

    FeaturedSentence sent = (FeaturedSentence) malloc(sizeof (struct FeaturedSentence));
    check_mem(sent);

    sent->words = DArray_create(sizeof (Word), 10);
    check_mem(sent->words);
    /*
    sent->scode = DArray_create(sizeof(float*), 10);
    check_mem(sent->scode);
    sent->postags = DArray_create(sizeof(char*), 10);
    check_mem(sent->postags);

    sent->parents = DArray_create(sizeof(int), 10);
    check_mem(sent->parents);
     */
    sent->length = 0;
    //sent->noscode = 0;
    //sent->scode_length = embedding_len;
    sent->feature_matrix_ref = NULL;
    sent->adjacency_matrix = NULL;

    return sent;

error:
    log_err("Sentence allocation error.");
    exit(1);
}



// TODO: Complete implementation

void free_FeaturedSentence(CoNLLCorpus corpus, int sentence_id) {
}

static void FeaturedSentence_check_and_add(CoNLLCorpus corpus, FeaturedSentence sent) {
    if (corpus->hasembeddings) {
        long sentence_embedding_length = -1;
        for (int i = 0; i < DArray_count(sent->words); i++) {

            Word w = (Word) DArray_get(sent->words, i);

            check(sentence_embedding_length == -1 || sentence_embedding_length == w->embedding->true_n,
                    "Multiple embedding lengths in a sentence. Current sentence has %ld. Previously seen has %ld", w->embedding->true_n, sentence_embedding_length);

            if (sentence_embedding_length == -1)
                sentence_embedding_length = w->embedding->true_n;
        }

        if (Root->embedding == NULL) {
            Root->embedding = vector_create(sentence_embedding_length);
        }

        if (corpus->embedding_length == -1) {
            corpus->embedding_length = sentence_embedding_length;
            
            // TODO: Be able to calculate this based on parameter...
            corpus->transformed_embedding_length = corpus->embedding_length * 6 + 6 ;
            if(corpus->embedding_transform == QUADRATIC)
                corpus->transformed_embedding_length = ((corpus->transformed_embedding_length) * (corpus->transformed_embedding_length+3))/2;

            log_info("Corpus has an embedding length of %ld (%ld with transformation)", sentence_embedding_length,corpus->transformed_embedding_length);
        } else {
            check(sentence_embedding_length == corpus->embedding_length,
                    "Sentence having embedding size of %ld conflicts with the rest of the corpus embedding size %ld", 
                    sentence_embedding_length, corpus->embedding_length);
        }
        
        if (corpus->feature_matrix_singleton == NULL){
            corpus->feature_matrix_singleton = FeatureMatrix_create(MAX_SENT_LENGTH, corpus->transformed_embedding_length,false);
        }
    }

    DArray_push(corpus->sentences, sent);
    
    debug("One more sentence is added into corpus...");

    return;
error:
    log_err("FeaturedSentence check has failed...");
    exit(1);
}

static DArray* find_corpus_files(const char *dir, DArray* sections) {
    struct dirent *entry;
    DIR *dp;

    DArray *array = DArray_create(sizeof (char*), 100);

    check(array != NULL, "Corpus file array creation failed.");

    char path[255];
    for (int i = 0; i < DArray_count(sections); i++) {
        int section = *((int*) DArray_get(sections, i));
        sprintf(path, "%s/%02d", dir, section);

        dp = opendir(path);
        check(dp != NULL, "Directory access error %s", path);

        while ((entry = readdir(dp))) {
            if (endswith(entry->d_name, ".dp")) {
                char *fullpath = (char*) malloc(sizeof (char) * (strlen(dir) + 4 + strlen(entry->d_name) + 1));
                check_mem(fullpath);
                sprintf(fullpath, "%s/%02d/%s", dir, section, entry->d_name);

                DArray_push(array, fullpath);
            }
        }

        closedir(dp);
    }

    return array;

error:
    log_err("Terminating...");
    exit(1);
}

void read_corpus(CoNLLCorpus corpus, bool build_feat_matrix) {
    DArray* files = find_corpus_files(corpus->base_dir, corpus->sections);

    char *line = NULL;
    size_t len = 0;

    FeaturedSentence sent = FeatureSentence_create();

    for (int i = 0; i < DArray_count(files); i++) {
        ssize_t read;
        char *file = (char*) DArray_get(files, i);

        FILE *fp = fopen(file, "r");
        check_mem(fp);

        while ((read = getline(&line, &len, fp)) != -1) {

            if (strcmp(line, "\n") != 0) {
                Word w = parse_word(line, corpus->hasembeddings);

                add_word(sent, w);

            } else {

                FeaturedSentence_check_and_add(corpus, sent);

                if (build_feat_matrix)
                    set_FeatureMatrix(NULL, corpus, DArray_count(corpus->sentences) - 1);

                sent = FeatureSentence_create();
            }

        }

        fclose(fp);
    }

    free(line);

    // DArray_clear_destroy(files);

    log_info("Total of %d sentences", DArray_count(corpus->sentences));


    return;
error:
    log_err("Terminating...");
    exit(1);

}

void free_CoNLLCorpus(CoNLLCorpus corpus) {

    for (int si = 0; si < DArray_count(corpus->sentences); si++) {
        //FeaturedSentence sent = (FeaturedSentence)DArray_get(sentences, i);

        free_FeaturedSentence(corpus, si);
    }

    DArray_clear_destroy(corpus->disrete_patterns_parts);
    DArray_clear_destroy(corpus->embedding_pattern_parts);
}