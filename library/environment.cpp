#include "environment.h"

environment::environment()
{
    //curBudget_ = 0;
    algo_ = ml::LINEAR_REGRESSION;
    budget_ = 1;
}

void environment::read_from_stream(vw* model, istream &fin)
{
    string line;
    while (true)
    {
	getline(fin, line, '\n');
	if ( line.empty())
	    break;
	item tmp;
	tmp.init_example(model, &line[0]);
	items_.push_back(tmp);
    }
}


void environment::linear_regression(vw* model, int numPasses)  
{
    for (int ct = 0; ct < numPasses; ct++)
	{ 
	    //   cout << "Passes " << ct << endl;
	    for (int i = 0; i < items_.size(); i++)
		{
		    //  cout << "regress on " << i <<endl;
		    items_[i].vw_create_example(model);
		    model->learn(items_[i].ex);
		    items_[i].release_vwexample(model);
		    //   cout << "train " << ((label_data*) items_[i].ex->ld)->label << " ::: " << items_[i].ex->final_prediction << endl;
		}
	}
}

void environment::svm_rank(vw* model, int numPasses)
{
    cout << "start svm_rank binary classification" <<endl;
    for (int ct = 0; ct < numPasses; ct++)
	{
	    for (int i = 0; i < items_.size(); i++)
		{
		    for (int j = 0; j < items_.size(); j++)
			if (i != j)
			    {
			    
				//	cout << i << ";" << j <<endl;
				//	cout << "progress " << double(i)/items_.size() <<endl;
				char str[charN];
				// char *str = new char[charN];
				//  str = NULL;
				//  strcpy(str, items_[i].get_diff_str(items_[j]));
				items_[i].get_diff_str(items_[j], str);
				if (!strcmp(str, "\0")) 
				    {
					//			cout << "label equal" <<endl;
					continue;
				    }
			    
				// items_[i].get_diff_str(items_[j], str);
				//	cout <<"str:" << str <<endl;
				example *ex = VW::read_example(*model, str);
				model->learn(ex);
				//				cout << "learn" <<endl;
				VW::finish_example(*model, ex);
				//cout << "free example" <<endl;
				// delete []str;
			    }
		    //   items_[i].release_vwexample(model);
		}
	}
}

double environment::get_dis2set(item & itm)
{
    //return sum of distance
    double sumDis = 0;
    for (int i = 0; i < predList_.size(); i++)
	{
	    sumDis += itm.compute_simDis(items_[predList_[i]]);
	}
       return sumDis;
}
vector<double> environment::getVecSimVals()
{
    //return the normalized inverse sum of distance similarity
    double sumDisVal = 0;
    double minDisVal = 0;
    vector<double> recSumDists;
    double maxSumDis = -1e+9;
    for (int i = 0; i < items_.size(); i++)
	{
	    //sum of dis
	    sumDisVal = get_dis2set(items_[i]);
	    recSumDists.push_back(sumDisVal);
	    maxSumDis = (maxSumDis > sumDisVal)? maxSumDis: sumDisVal;
	}
    
    for (int i = 0; i < items_.size(); i++)
	{  
	    recSumDists[i] = 1 - recSumDists[i]/maxSumDis;
	}
    
    return recSumDists;
}
void environment::updateSimFeatures(vw* model, submodOracle & fOracle)
{
    //get the submodular values for things in the predList_
    // double predListScore = 0;
    //  cout << "updateFeatures" << endl;    
    vector<simpleItem> predListItems;
    for (int i = 0; i < predList_.size(); i++)
	{
	    simpleItem tmpItem;
	    tmpItem.itemId = predList_[i];
	    tmpItem.envId = items_[tmpItem.itemId].get_envId();
	    tmpItem.label = items_[tmpItem.itemId].get_label();
	    predListItems.push_back(tmpItem);
	}

    vector<double> vecSumSim = getVecSimVals();
    double newLabel = 0;
    for (int i = 0; i < items_.size(); i++)
	{
	    vector<feat_pair> vec_feat;
	    feat_pair featPair;
	    featPair.featId = featIndex::SUMSIM;  
	    featPair.val = vecSumSim[i];
	    //    cout << "Similarity for item " << i << " to predList :" << featPair.val << endl;

	    vec_feat.push_back(featPair);
	
	    // compute marginal gain and update label  
	    simpleItem tmpItem;
	    tmpItem.itemId = i;
	    tmpItem.envId = items_[tmpItem.itemId].get_envId();
	    tmpItem.label = items_[tmpItem.itemId].get_label();
	    newLabel = fOracle.getMarginalScore(predListItems, tmpItem);
	    //    cout << "newLabel " <<newLabel << endl;
	    items_[i].update_example(model, vec_feat, newLabel, true); 
	    //need to check whether budget overflow
	}
}

void environment::updateSimFeaturesPredictOnly(vw* model)
{
    vector<double> vecSumSim = getVecSimVals();
    for (int i = 0; i < items_.size(); i++)
	{
	    vector<feat_pair> vec_feat;
	    feat_pair featPair;
	    featPair.featId = featIndex::SUMSIM;  
	    featPair.val = vecSumSim[i];
	    vec_feat.push_back(featPair);

	    // no need to update label, cuz it's prediction only  
	    items_[i].update_example(model, vec_feat); 
	    //need to check whether budget overflow
	}
}

void environment::oneRoundTrain(vw* model, int numPasses)
{
    //cout <<"startOneRoundTraining" <<endl;
    model->training = true; 
    if (algo_ == ml::LINEAR_REGRESSION)
	{
	    linear_regression(model, numPasses);
	}
    else if (algo_ == ml::SVM_RANK)
	{
	    svm_rank(model, numPasses);
	}
}

void environment::oneRoundPrediction(vw* model, bool useGreedyOracle )
{
    //cout << "startoneRoundPrediction" <<endl;
    double maxValue = -1e10;
    int maxItemId = -1;
    double modelPredScore = 0;
    model->training = false;  
    haveSpace_ = false;
    for (int i = 0; i < items_.size(); i++)
	{
	    if (!useGreedyOracle)  //if use model prediction
		{
		    //vw will produce warning here because it will see non-binary labels, but for prediction, label does not matter
		    items_[i].vw_create_example(model);   
		    if (algo_ == ml::SVM_RANK)
			((label_data*) items_[i].ex->ld)->label = 1;
		    model->learn(items_[i].ex);
		    modelPredScore = items_[i].ex->final_prediction;
		    items_[i].release_vwexample(model);
		}
	    else  //if use greedy ground truth, note label is updated as marginal gain
		{
		    // modelPredScore  = ((label_data*) items_[i].ex->ld)->label;
		    modelPredScore = items_[i].get_label();
		}
	    //    cout << "predictStatus for Item " << i << ": gdScore " << ((label_data*) items_[i].ex->ld)->label << " vwScore " << items_[i].ex->final_prediction <<endl;
	    if ( curBudget_ + items_[i].size > budget_ )
		{
		    tag_usable[i] = false;
		}
	    if ( modelPredScore > maxValue && tag_usable[i])  //find largest 
		{
		    maxValue = modelPredScore;
		    maxItemId = i;
		    if (curBudget_ + items_[i].size < budget_)
			haveSpace_ = true;
		}
	}
    if (maxItemId != -1)
	{
	    predList_.push_back(maxItemId);
	    curBudget_ += items_[maxItemId].size; 
	    //both for cardinality and knapsack constraint
	    curSubModScore_ += maxValue * items_[maxItemId].size;  
	    tag_usable[maxItemId] = false;
	}
    /*
    //display prediction information
    for (int i = 0; i < predList_.size(); i++)
	{
	    cout << predList_[i] << ",";
	}
    cout << " currentSubModScore:  " << curSubModScore_ << endl;
    cout << " currentBudget: " << curBudget_<< endl;
    */
}

void environment::multiRoundTrain(vw* model, submodOracle & fOracle, bool useGreedyPolicy, int numPasses)
{
    curBudget_ = 0;
    curSubModScore_ = 0;
    haveSpace_ = true;
    tag_usable = new bool[items_.size()];
    memset(tag_usable, true , sizeof(bool) * items_.size());
    while (haveSpace_)
	{
	    //	    cout << "trainingOneRound" << endl;
	    oneRoundTrain(model, numPasses);
	    //if useGreedyPolicy is true 
	    oneRoundPrediction(model, useGreedyPolicy); 
	    updateSimFeatures(model, fOracle);
	}
    
     getPerSlotScore(fOracle);
     //  release_items_vwexample(model);
}

void environment::multiRoundPredictOnly(vw* model)
{
    curBudget_ = 0;
    curSubModScore_ = 0;
    haveSpace_ = true;
    tag_usable = new bool[items_.size()];
    memset(tag_usable, true , sizeof(bool) * items_.size());
    while (haveSpace_)
	{
	    oneRoundPrediction(model); 
	    updateSimFeaturesPredictOnly(model);
	} 
    //   release_items_vwexample(model);
}

vector<double> environment::getPerSlotScore( submodOracle & fOracle)
{
    vector<simpleItem> predListItems;
    double gain = 0 ;
    double fVal = 0;
    vector<double> perSlotScore;
    for (int pos = 0; pos < predList_.size(); pos++)
	{
	    //create partial list
	    predListItems.clear();
	    for (int i = 0; i < pos; i++)
		{
		    simpleItem tmpItem;
		    tmpItem.itemId = predList_[i];
		    tmpItem.envId = items_[tmpItem.itemId].get_envId();
		    tmpItem.label = items_[tmpItem.itemId].get_label();
		    predListItems.push_back(tmpItem);
		}
	    simpleItem tmpItem;
	    tmpItem.itemId = predList_[pos];
	    tmpItem.envId = items_[tmpItem.itemId].get_envId();
	    tmpItem.label = items_[tmpItem.itemId].get_label();
	    gain = fOracle.getMarginalScore(predListItems, tmpItem)  ;
	    //	    cout << "PredictList at Pos " << pos << " have marginal gain : " << gain <<endl;
	    fVal += gain;
	    perSlotScore.push_back(fVal);
	    //cout << "PredictList at Pos " << pos << " is " << predList_[pos] << " , has submod value : " << fVal << endl;
	}
    return perSlotScore;
}

void environment::release_items_vwexample(vw* model)
{
    for (int i = 0; i < items_.size(); i++)
	{
	    items_[i].release_vwexample(model);
	}
}

