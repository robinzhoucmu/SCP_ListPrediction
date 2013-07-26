#ifndef ITEM_H
#define ITEM_H

#include "../vowpalwabbit/vw.h"
#include "../vowpalwabbit/parser.h"
#include "../vowpalwabbit/simple_label.h"
#define charN 1000

using namespace std;
struct feat_pair  
{
    int featId;   //assume input featId is integer
    double val;
};
class item
{
    public:
    
    item();
    void read_from_string(string str);
    //passing the updatedDynamicFeatures and it will do the matching, assume featId is increasing
    void release_vwexample(vw* model);
    void update_dynamicSimFeature(vector<feat_pair> & vecFeat2 );
    void parse_vw_input (char *line);
    void init_example(vw*model,  char *line);
    void update_example(vw*model, vector<feat_pair> & vecFeat2, double newLabel = 0, bool updateLabel = false); 
    void build_example_from_pairdiff(vw*model, const item& a, const item& b);
    void vw_create_example(vw*model){ex = VW::read_example(*model, expanded_line_);};
    void get_diff_str(item & itm, char * diff_line);
    vector<feat_pair> & get_vec_sim_feat(){
	return vec_sim_feat_;
    }
    vector<feat_pair> & get_vec_quality_feat(){
	return vec_quality_feat_;
    }
    vector<feat_pair> & get_vec_dynamicSim_feat(){
	return vec_dynamicSim_feat_;
    }
    double & get_label(){
	return label_;
    }
    int & get_envId(){
	return envId_;
    }
    double compute_simDis ( item & itm);
    example *ex;
    double size;

    private:
    
    char label_line_[100];  //the string that stores label information
    char prefix_line_[charN]; //the string that stores quality features, would not change after first parsing
    char expanded_line_[charN];  //the complete string, dynamically changing
    double label_;
    double importance_;
    
    int envId_;
    int dimAllFeature_;
    int dimQualityFeature_;
    double marginalGain_;
    vector<feat_pair> vec_quality_feat_;
    //user specified similiary feature, e.g., tf-idf vector, or can be same as quality features
    vector<feat_pair> vec_sim_feat_;   
    //dynamic similiarity features computed on the fly
    //quadratic part is handled using VW -q
    vector<feat_pair> vec_dynamicSim_feat_;
    //  void parse_label(char *start,  double & label, double &weight, int &envId);
    void parse_label(const char* start);
    void construct_feat_vec(char *start, char*end, vector<feat_pair> & vec_feat);
    void init_dynamicSim_feat_vec();
    vector<feat_pair>   get_diff_vec_feat(vector<feat_pair> & vec1, vector<feat_pair> & vec2 );
    char* construct_feat_line(vector<feat_pair> & vec_feat); 
    //build the expanded_line for initializing example
    void construct_expanded_line(); //build a new string incoporating update of label and dynamicSimFeatures
    void construct_label_line();
};
    

#endif
