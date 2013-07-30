#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <unistd.h>
#include <assert.h>
#include <ctime>
#include <cstdlib>
//#include <environment.h>
#include "seqMachine.h" 
#include <boost/program_options.hpp>

#include "../vowpalwabbit/vw.h"

using namespace std;
namespace po = boost::program_options;


int main(int argc, char * argv[])
{  
    std::srand ( unsigned ( 0 ) ); 
    string vwparams = " -f predictor2.vw  --readable_model predictorInfo.txt";
    string train_file_name, test_file_name;
    int train_num_iters = 5;
    int train_num_passes = 1;
    double budget = 8;
    double l2Lambda = 0;
    double learningRate = 0.5;
    ml::algorithm_t algo = ml::LINEAR_REGRESSION;
    string algo_name;
    po::variables_map vm;
    po::options_description desc("Allowed options");
    
    desc.add_options()
	// ("vwparams", po::value<string>(&vwparams), "vw parameters for model instantiation (-i model ...)")
	("data,d", po::value<string>(&train_file_name), "input file for training")
	("iters", po::value<int>(&train_num_iters), "num of iterations, default is 5" )
	("passes", po::value<int>(&train_num_passes), "num of passes of data, default is 10" )
	("test,t", po::value<string>(&test_file_name), "test file for prediction")
	("budget,b", po::value<double>(&budget), "budget" )
	("l2", po::value<double>(&l2Lambda), "lambda term multiplier for l2 norm, default is 0")
	("lr", po::value<double>(&learningRate), "const learning rate, default is 0.5")
	("algo", po::value<string>(&algo_name), "specify learning model, choose between linear_regression or svm_rank. By default is linear_regression")
	;
     try {
                po::store(po::parse_command_line(argc, argv, desc), vm);
                po::notify(vm);
        }
     catch(exception & e)
	 {
	     cout << endl << argv[0] << ": " << e.what() << endl << endl << desc << endl;
	     exit(2);
	 }

     if (vm.count("help")) {
	 cout << desc << "\n";
	 return 1;
     }
     if (vm.count("algo"))
	 {
	     if (!algo_name.compare("svm_rank"))
		 {
		     algo = ml::SVM_RANK;
		 }
	     else
		 {
		     algo = ml::LINEAR_REGRESSION;
		 }
	 }
     
     
     seqMachine testseqMachine;
     multipleGuess01 fOracle;
    
     testseqMachine.setBudget(budget);
     testseqMachine.setTrainingParameters(train_num_iters, train_num_passes, learningRate, l2Lambda, algo);
     testseqMachine.scp_train( fOracle, train_file_name);
     
     //testing 
     cout << "----------" <<endl;     
     testseqMachine.scp_predict( test_file_name);
     testseqMachine.check_predict_score(fOracle);
     
    return 0;
}
