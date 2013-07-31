#ifndef SEQMACHINE_H
#define SEQMACHINE_H
#include "environment.h"
//#include <stdlib.h>
//#include <stdio.h>
//#include "globalheaders.h"
using namespace std;

class seqMachine
{
  public:
    
    seqMachine();
    void setBudget(double budget){budget_ = budget;}
    void setAlgoType(ml::algorithm_t algo = ml::LINEAR_REGRESSION)
    {
	algo_ = algo;
    }
    void setTrainingParameters(int num_iters = 5, int num_passes = 5, double learningRate = 0.5, double l2Lambda = 0)
    {	
	num_iters_ = num_iters; 
	num_passes_ = num_passes;
	learningRate_ = learningRate;
	l2Lambda_ = l2Lambda;

    }

    //assume the each env_items set is separated by an empty line
    void construct_envs_items(vw*model, istream &fin);  
    void one_iter_train(vw* model, submodOracle & fOracle, istream &fin, bool isGreedy = false);
    void multiple_pass_from_cache(vw* model);
    void scp_train( submodOracle & fOracle, string fileName); //num_iters
    void scp_predict(string fileName);
    void cross_validation( string trainingFileName, string validationFileName);//model need to be initialized
    double get_predict_score (submodOracle & fOracle);

 private:
    
    void initialize_vw_training_model();
    void initialize_vw_testing_model();
    string double2str(double val){
	std::ostringstream out;
	out << std::fixed << val;
	return out.str();
    }
    string int2str(int number)
    {
	stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
    }

    vw* model;
    vector<int> generate_random_index(int tot);
    vector<environment> envs;
    int num_iters_;  //including the initial greedy policy, if ==1 then exact imitation of greedy
    int num_passes_;
    double budget_;
    double learningRate_;
    double l2Lambda_;
    ml::algorithm_t algo_;
    

    
   
};

#endif
