#ifndef GLOBALHEADERS
#define GLOBALHEADERS
#define REGRESSION 1
namespace ml{
    typedef enum algorithm
    {
	LINEAR_REGRESSION = 100,
	SVM_RANK = 200,
    } algorithm_t;
}

namespace featIndex{
    typedef enum featIndex
    {
	SUMSIM = 1,
	AVGSIM = 2,
    }featIndex_t;
}
#endif
