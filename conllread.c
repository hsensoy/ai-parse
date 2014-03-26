#include <stdio.h>

#include "corpus.h"
#include "debug.h"
#include "dependency.h"
#include "util.h"

void print_FeaturedSentence(FeaturedSentence sent){
	
	for(int i = 0; i < DArray_count(sent->words);i++){
		Word w = (Word)DArray_get(sent->words,i);
		
		printf("%d\t%s\t%s\t%d\t", w->id,w->form,w->postag,w->parent);
		
		for(int j = 0 ; j < w->embedding->true_n;j++){
			printf("%f ", w->embedding->data[j]);
		}
		
		printf("\n");
		
	}
	
	for(int _from = 0; _from <= DArray_count(sent->words); _from++)
		for(int _to = 1; _to <= DArray_count(sent->words); _to++){
			
			if(_to != _from){
			
			vector embedding = (sent->feature_matrix_ref->matrix_data)[_from][_to]->continous_v;
			
			vprint(embedding);
				
			printf("\n");
		
			}
		
		}
	
}

int __main(int argc, char **argv)
{
	
	DArray *sect = range(2,5);	//range(2,22);
	DArray *test_sec = range(22,23);
	
	const char* corpus1_dir = "/Users/husnusensoy/uparse/data/nlp/treebank/treebank-2.0/combined/conll";
	const char* corpus2_dir = "/Users/husnusensoy/uparse/data/nlp/treebank/treebank-2.0/combined/conll_scodewiki";
	/*
	log_info("Reading corpus %s without embeddings",corpus1_dir);
	DArray *conll_corpus = create_conllcorpus(corpus1_dir, sect, false) ;
	check(DArray_count(conll_corpus) == 39832, "Expeceted number of sentence in corpus was 39832 whereas got %d",DArray_count(conll_corpus));
	
	free_conllcorpus(conll_corpus);
	
	log_info("Reading corpus %s with embeddings",corpus1_dir);
	conll_corpus = create_conllcorpus(corpus1_dir, sect, true) ;
	
	check(DArray_count(conll_corpus) == 39832, "Expeceted number of sentence in corpus was 39832 whereas got %d",DArray_count(conll_corpus));
	free_conllcorpus(conll_corpus);
	
	///
	log_info("Reading corpus %s without embeddings",corpus2_dir);
	conll_corpus = create_conllcorpus(corpus2_dir, sect, false) ;
	check(DArray_count(conll_corpus) == 39832, "Expeceted number of sentence in corpus was 39832 whereas got %d",DArray_count(conll_corpus));
	
	free_conllcorpus(conll_corpus);
	
	
	log_info("Reading corpus %s with embeddings",corpus2_dir);
	DArray * conll_corpus = create_conllcorpus(corpus2_dir, sect, true) ;
	
	check(DArray_count(conll_corpus) == 39832, "Expeceted number of sentence in corpus was 39832 whereas got %d",DArray_count(conll_corpus));
	free_conllcorpus(conll_corpus);
	*/
	/*
	log_info("Reading corpus %s with embeddings",corpus2_dir);
	DArray * conll_corpus = create_conllcorpus(corpus2_dir, sect, true) ;
	
	check(DArray_count(conll_corpus) == 39832, "Expeceted number of sentence in corpus was 39832 whereas got %d",DArray_count(conll_corpus));
	free_conllcorpus(conll_corpus);
	
	DArray_clear_destroy(sect);
	 */

	const char* EMBEDDING_PATTERN = "p-1v_p0v_p1v_c-1v_c0v_c1v_tl";
	log_info("Creating CoNLL Corpus meta for %s",corpus2_dir);
	CoNLLCorpus training = create_CoNLLCorpus(corpus2_dir,sect,25,EMBEDDING_PATTERN,QUADRATIC, NULL);
	
	log_info("Creating CoNLL Corpus meta for %s",corpus2_dir);
	CoNLLCorpus test = create_CoNLLCorpus(corpus2_dir,test_sec,25,EMBEDDING_PATTERN,QUADRATIC, NULL);
	
	log_info("\tReading training corpus into memory");
	read_corpus(training, false);
	
	log_info("\tReading test corpus into memory");
	read_corpus(test, false);
	
	set_FeatureMatrix(NULL,training, 0);
	
	//FeaturedSentence sent = (FeaturedSentence)DArray_get(training->sentences,0);
	
	//print_FeaturedSentence(sent);
	
	
	//build_adjacency_matrix(training,0,w,NULL);
	
	
	PerceptronModel model = PerceptronModel_create(training, NULL);
	log_info("\tStarting to parse corpus");
	//train_perceptron_parser(model, training, 1);
	float *numit_avg = (float*)malloc(sizeof(float)* 3);

	for (int numit = 1; numit <= 10 ; numit++) {
        log_info("Testing numit %d",numit);
        
        //train_perceptron_parser(model, training, numit);
        
        train_perceptron_once(model,training,-1);
        
        log_info("Parser training done");
            
        double accuracy = test_perceptron_parser(model, test, true, true);
        
        
        log_info("\n\tnumit=%d avg=%lf stddev=%lf\n", numit, accuracy, 0.0);
        
        numit_avg[numit-1] = accuracy;
    }
	


	
	//train_perceptron_once(model, training);
		
	/*
	PerceptronModel model = PerceptronModel_create(training, NULL);
	
	log_info("\tStarting to parse corpus");
	train_perceptron_once(model, training);
	 */
	 log_info("Weights of perceptron...");
	 for(int i = 0 ; i< model->embedding_w->true_n ;i++ ){
		 printf("w[%d] = %f\n",i,(model->embedding_w->data)[i]);
		 
	 }
	 
	for (int i = 0; i < 10; i++) {
        log_info("%d\t%f\t%f",i+1,numit_avg[i],0.0);
    }
	 
	return 0;
}
