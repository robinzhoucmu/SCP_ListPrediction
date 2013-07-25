#include "submodOracle.h"



multipleGuess01::multipleGuess01()
: submodOracle()
{
}

double multipleGuess01::queryScore(vector<simpleItem> & vecItems)
{
    // double sum = 0;
    for (int i = 0; i < vecItems.size(); i++)
	{
	    if (vecItems[i].label == 1)
		return 1;
	}
    return 0;
}

double multipleGuess01::getMarginalScore(vector<simpleItem> & vecItems, simpleItem & appItem)
{
    double prevScore = queryScore(vecItems);
    //   cout << "prevScore" << prevScore << endl;
    vecItems.push_back(appItem);
    double appScore = queryScore(vecItems);
    //    cout << "appScore" << appScore << endl;
    vecItems.pop_back();
    return appScore - prevScore;
}
