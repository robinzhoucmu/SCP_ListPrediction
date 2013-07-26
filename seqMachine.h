#ifndef SEQMACHINE_H
#define SEQMACHINE_H
#include "environment.h"
//#include "globalheaders.h"
using namespace std;

class seqMachine
{
  public:
    
    // seqMachine(int num_iters = 3, int num_passes = 10, double budget = 1);
    seqMachine();
    void setBudget(double budget){budget_ = budget;}
    void setTrainingParameters(int num_iters = 3, int num_passes = 10, ml::algorithm_t algo = ml::LINEAR_REGRESSION)
    {	
	num_iters_ = num_iters; 
	num_passes_ = num_passes;
	algo_ = algo;
    }

    //assume the each env_items set is separated by an empty line
    void construct_envs_items(vw*model, istream &fin);  
    void scp_train(vw*model, submodOracle & fOracle, string fileName); //num_iters
    void scp_predict(vw* model, string fileName);
    void check_predict_score (submodOracle & fOracle);
 private:

    vector<int> generate_random_index(int tot);
    vector<environment> envs;
    int num_iters_;  //including the initial greedy policy, if ==1 then exact imitation of greedy
    int num_passes_;
    double budget_;
    ml::algorithm_t algo_;
    

    
   
};

#endif
