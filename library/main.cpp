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

string double2str(double val)
{
    std::ostringstream out;
    out << std::fixed << val;
    return out.str();
}

int main(int argc, char * argv[])
{  
    // string vwparams = " -q qd  -f predictor2.vw  --readable_model predictorInfo.txt";
   std::srand ( unsigned ( 0 ) ); 
    string vwparams = " -f predictor2.vw  --readable_model predictorInfo.txt";
    string train_file_name, test_file_name;
    int train_num_iters = 5;
    int train_num_passes = 10;
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
     
     //construct vwparams
      vwparams = " -q qd -f predictor.vw --readable_model predictorInfo.txt";
      if (algo == ml::SVM_RANK)
	  {
	      vwparams += " --loss_function hinge ";
	  }
     //append l2 lambda norm
     vwparams += (" --l2 " + double2str(l2Lambda)); 
     //append learning rate
     vwparams += ((" -l ") + double2str(learningRate));
     vw* model = VW::initialize(vwparams);
     seqMachine testseqMachine;
     multipleGuess01 fOracle;

     testseqMachine.setBudget(budget);
     testseqMachine.setTrainingParameters(train_num_iters, train_num_passes, algo);
     testseqMachine.scp_train(model, fOracle, train_file_name);
     VW::finish(* model);
    
     //test on the training data first
     cout << "----------" <<endl;
     string vwparams_test = "-t -i predictor.vw -p predictionTrained.txt ";
     vw* model_test = VW::initialize(vwparams_test);
     testseqMachine.scp_predict(model_test, test_file_name);
     testseqMachine.check_predict_score(fOracle);
     VW::finish(* model_test);
    return 0;
}
