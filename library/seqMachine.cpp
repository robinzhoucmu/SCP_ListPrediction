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

void seqMachine::scp_train(vw*model, submodOracle & fOracle, string fileName)
{
    
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
}

void seqMachine::scp_predict(vw* model, string fileName)
{
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
}

void seqMachine::cross_validation(vw * model, string trainingFileName, string validationFileName) 
{
    double validationAcc = 0;
    int best_num_iters, best_num_passes;
    double best_learning_rate;
    double l2Lambda;
    double candidate_learning_rate[4] = {0.1, 0.5, 2, 10};
    double candidate_num_passes[4] = {1, 2, 5, 10};
    double candidate_l2Lambda[3] = {0.00001, 0.0005, 0.005};
    int max_iters = 10;
    for (int ind_lr = 0; ind_lr < sizeof(candidate_learning_rate)/sizeof(candidate_learning_rate[0]); ind_lr++)
	{
	    for (int ind_nPasses = 0; ind_nPasses < sizeof(candidate_num_passes)/sizeof(candidate_num_passes[0]); ind_nPasses++)
		{
		    for (int ind_l2Lambda = 0; ind_l2Lambda < sizeof(candidate_l2Lambda)/sizeof(candidate_l2Lambda[0]); ind_l2Lambda++)
			{
			    
			}
		}
	}

}

void seqMachine::check_predict_score (submodOracle & fOracle)
{
    double avgFinalScore;
    for (int i = 0; i < envs.size(); i++)
	{
	    //	    cout << "For Environment " << i <<endl;
	    vector<double> score = envs[i].getPerSlotScore(fOracle);
	    avgFinalScore += score[score.size() - 1];
	}
    avgFinalScore /= envs.size();
    cout << "Average Prediction Final Score " << avgFinalScore << endl;
}

vector<int> seqMachine::generate_random_index(int tot)
{
    vector<int> randIndex;
    for (int i = 0; i < tot; i++ )
	randIndex.push_back(i);
    std::random_shuffle(randIndex.begin(), randIndex.end());
    /*
    for (int i = 0; i < randIndex.size(); i++)
	{
	    cout << randIndex[i] << ",";
	}
    cout << endl;
    */
    return randIndex;
}

void seqMachine::initialize_vw_training_model(vw* model)
{
    
}

void seqMachine::initialize_vw_testing_model(vw *model)
{

}
