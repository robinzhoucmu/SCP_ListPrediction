#ifndef SUBMODORACLE_H
#define SUBMODORACLE_H
#include "item.h"

using namespace std;

struct simpleItem
{
    int itemId;
    int envId;
    double label;
};

class submodOracle
{
 public:
    submodOracle(){};
    virtual ~submodOracle(){};
    virtual double queryScore(vector<simpleItem> & vecItems){return 0;};
    //getMarginalScore returns marginal score for cardinality constraint and 
    //normalized marginal score for knapsack constraint
    virtual double getMarginalScore(vector<simpleItem> & vecItems, simpleItem & appItem){return 0;};
    //  virtual double getLeftBudget(vector<simpleItem> & vecItems, simpleItem & appItem);
};

//below is multipleGuessLearning examples
//there is no need to read from file since the feature/item files will have the 0/1 label 
class multipleGuess01
: public submodOracle
{
 public:
    multipleGuess01();
    virtual ~multipleGuess01(){};
    virtual double queryScore(vector<simpleItem> & vecItems);
    virtual double getMarginalScore(vector<simpleItem> & vecItems, simpleItem & appItem);
    // double getLeftBudget(vector<simpleItem> & vecItems, simpleItem & appItem);
 private:
    
};

#endif
