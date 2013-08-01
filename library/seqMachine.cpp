#include "seqMachine.h"

seqMachine::seqMachine() 
{
    /*
    num_iters_ = num_iters;
    num_passes_ = num_passes;
    budget_ = budget;
    */
}

void seqMachine::construct_envs_items(vw*model, istream &fin)
{
    int id, numItems;
    //   char str[100];
    string str;
    envs.clear();
    int ct = 0;
    while (!fin.eof())
	{
	    environment env;
	    cout << "read_from_stream for EnvId " << ct <<endl;
	    env.read_from_stream(model, fin);
	    envs.push_back(env);
	    ct++;
	}
}
void seqMachine::one_iter_train(vw* model, submodOracle & fOracle, istream &fin, bool isGreedy)
{
    int ct = 0;
    while (!fin.eof())
	{
	    environment env;	    
	    env.read_from_stream(model, fin);
	    env.setBudget(budget_);
	    env.setAlgoType(algo_);
	    env.multiRoundTrain(model, fOracle, isGreedy, num_passes_);
	    ct++;
	}
    //  multiple_pass_from_cache(model);
    cout << "readonly_one_iterations_train on " << ct << " environments" <<endl;
}

void seqMachine::multiple_pass_from_cache(vw* pointer)
{
    pointer->training = true;
    if (pointer->numpasses > 1)
	{
	    cout << "multipass" << endl;
	    adjust_used_index(*pointer);
	    pointer->do_reset_source = true;
	    VW::start_parser(*pointer,false);
	    pointer->l.drive(pointer);
	    VW::end_parser(*pointer); 
	}
    else
	release_parser_datastructures(*pointer);
}

void seqMachine::scp_train( submodOracle & fOracle, string fileName)
{
    initialize_vw_training_model();
    //initialize with greedy on each environment
    cout <<"start SCP_TRAIN by Greedy Coaching" <<endl;
    vector<int> randIndex;
    ifstream fin;
    fin.open(fileName.c_str());
    one_iter_train(model, fOracle, fin, true);
    fin.close();
    //    construct_envs_items(model, fin);
    /*
    envs.clear();
    int ct = 0;
    while (!fin.eof())
	{
	    environment env;	    
	    //  cout << "read_from_stream for EnvId " << ct <<endl;
	    env.read_from_stream(model, fin);
	    env.setBudget(budget_);
	    env.setAlgoType(algo_);
	    envs.push_back(env);
	    //   cout << "numItems " << envs[ct].get_numItems() <<endl;
	    //  envs[ct].multiRoundTrain(model, fOracle, true, num_passes_);
	    ct++;
	}
    
    //test whether can back up one copy of all enviroments
    randIndex = generate_random_index(envs.size());
    for (int i = 0; i< envs.size(); i++)
	{
   	    envs[randIndex[i]].multiRoundTrain(model, fOracle, true, num_passes_);
	}
    fin.close();
    */
    int ct  = 0;
    cout << "greedy coaching finished" <<endl;
    
    for (int iter = 1; iter < num_iters_; iter++)
	{
	    cout << "iterations " << iter << endl;
	    envs.clear();
	    ct = 0;
	    fin.open(fileName.c_str());
	    one_iter_train(model, fOracle, fin, false);
        
	    /*
	    while (!fin.eof())
		{
		    environment env;	    
		    // cout << "read_from_stream for EnvId " << ct <<endl;
		    env.read_from_stream(model, fin);
		    env.setBudget(budget_);
		    env.setAlgoType(algo_);
		    envs.push_back(env);
		    //  cout << "numItems " << envs[ct].get_numItems() <<endl;
		    //   envs[ct].multiRoundTrain(model, fOracle, false, num_passes_);
		    ct++;
		}
	    
	    //    randIndex = generate_random_index(envs.size());
	    for (int i = 0; i< envs.size(); i++)
		{
		    envs[randIndex[i]].multiRoundTrain(model, fOracle, false, num_passes_);
		}
	    */	    
	    fin.close();
	}
    multiple_pass_from_cache(model);   
    cout << "training completed" << endl;
    VW::finish(*model);
}

void seqMachine::scp_predict(string fileName)
{
    initialize_vw_testing_model();
    cout << "Run SCP_Predict Only" <<endl;
    ifstream fin;
    fin.open(fileName.c_str());
    //    construct_envs_items(model, fin);
    // for (int i = 0; i < envs.size(); i++)
    int ct = 0;
    envs.clear();
    while (!fin.eof())	
	{
	    environment env;	    
	    //	    cout << "read_from_stream for EnvId " << ct <<endl;
	    env.read_from_stream(model, fin);
	    env.setBudget(budget_);
	    envs.push_back(env);
	    //	    cout << "numItems " << envs[ct].get_numItems() <<endl;
	    envs[ct].multiRoundPredictOnly(model);
	    ct++;
	}
    fin.close();
    VW::finish(*model);
}
//need to set algo type beforehand
void seqMachine::cross_validation( string trainingFileName, string validationFileName, submodOracle & fOracle) 
{
    cout << "start cross validation" << endl;
    double curScore = 0;
    double bestScore = 0;
    int best_num_iters, best_num_passes;
    double best_learning_rate, best_l2Lambda;
    int iters, passes;
    double l2Lambda, learning_rate;
    
    double candidate_learning_rate[4] = {0.1, 0.5, 2, 10};
    int candidate_num_passes[4] = {1, 2, 5, 10};
    double candidate_l2Lambda[3] = {0.00001, 0.0005, 0.005};
    int candidate_iterations[3] = {2, 5, 10};
    for (int ind_lr = 0; ind_lr < sizeof(candidate_learning_rate)/sizeof(candidate_learning_rate[0]); ind_lr++)
	{
	    for (int ind_nPasses = 0; ind_nPasses < sizeof(candidate_num_passes)/sizeof(candidate_num_passes[0]); ind_nPasses++)
		{
		    for (int ind_l2Lambda = 0; ind_l2Lambda < sizeof(candidate_l2Lambda)/sizeof(candidate_l2Lambda[0]); ind_l2Lambda++)
			{
			    for (int ind_iters = 0; ind_iters < sizeof(candidate_iterations)/sizeof(candidate_iterations[0]); ind_iters++)
				{
				    learning_rate =  candidate_learning_rate[ind_lr];
				    l2Lambda = candidate_l2Lambda[ind_l2Lambda];
				    iters = candidate_iterations[ind_iters];
				    passes = candidate_num_passes[ind_nPasses];

				    setTrainingParameters(iters, passes, learning_rate, l2Lambda);
				    scp_train(fOracle, trainingFileName);
				    scp_predict(validationFileName);
				    curScore = get_predict_score(fOracle);
				    if (curScore > bestScore)
					{
					    bestScore = curScore;
					    best_learning_rate = learning_rate;
					    best_l2Lambda = l2Lambda;
					    best_num_passes = passes;
					    best_num_iters = iters;
					    cout << bestScore << " , " << best_learning_rate << " , " << best_l2Lambda << " , " << best_num_passes << " , " << best_num_iters << endl;
					}
				}
			}
		}
	}

    //train the model with the best parameter
    cout << "bestScore: " << bestScore << endl;
    cout << "bestLearningRate: " << best_learning_rate << endl;
    cout << "bestL2Lambda: " << best_l2Lambda << endl;
    cout << "bestNumPasses" << best_num_passes << endl;
    cout << "bestNumIterations" << best_num_iters << endl;
    setTrainingParameters( best_num_iters, best_num_passes, best_learning_rate, best_l2Lambda );
    scp_train(fOracle, trainingFileName);

}

double seqMachine::get_predict_score (submodOracle & fOracle)
{
    cout << "predict on " << envs.size() << " environments" << endl;
    double avgFinalScore;
    for (int i = 0; i < envs.size(); i++)
	{
	    //	    cout << "For Environment " << i <<endl;
	    vector<double> score = envs[i].getPerSlotScore(fOracle);
	    avgFinalScore += score[score.size() - 1];
	}
    avgFinalScore /= envs.size();
    cout << "Average Prediction Final Score " << avgFinalScore << endl;
    return avgFinalScore;
}

vector<int> seqMachine::generate_random_index(int tot)
{
    vector<int> randIndex;
    for (int i = 0; i < tot; i++ )
	randIndex.push_back(i);
    std::random_shuffle(randIndex.begin(), randIndex.end());
    
    return randIndex;
}

void seqMachine::initialize_vw_training_model()
{
    string vwparams = "-k -q qd -f predictor.vw --readable_model predictorInfo.txt -c ";
    // add number of passes
    vwparams += " --passes " + int2str(num_passes_);
    // add l2 regularizer
    vwparams += " --l2 " + double2str(l2Lambda_);
    // add learningRate
    vwparams += " -l " + double2str(learningRate_);
    cout << vwparams << endl;
    model = VW::initialize(vwparams);
}

void seqMachine::initialize_vw_testing_model()
{
    string vwparams = "-t -i predictor.vw";
    model = VW::initialize(vwparams);
}
