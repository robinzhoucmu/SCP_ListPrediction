#include "item.h"
#include "globalheaders.h"
#include <cmath>
item::item()
{
    //  dimAllFeature_ = 0; 
    // dimQualityFeature_ = 0;
    
}
//assume that one line of feature
//score,f1,f2,f3,....
void item::read_from_string(string str) 
{
    
}
void item::release_vwexample(vw* model)
{
    VW::finish_example(*model, ex);
}
void item::parse_label(const char *start)
{
    sscanf(start, "%lf %lf %lf %d|", &label_, &importance_, &size, &envId_);
    // sscanf(start, "%lf %lf", &label, &weight);
    //     cout<<label_<<","<<importance_<<","<<envId_<<endl;
}

void item::construct_feat_vec(char *start, char*end, vector<feat_pair> & vec_feat)
{
    substring stmp = {start, end};
    v_array<substring> segments;
    tokenize(' ', stmp, segments);
    // index starts from 1, since the first segment would be the namespace
    vec_feat.clear();
    for (int i = 1; i < segments.size(); i++)
	{
	    feat_pair fpair;
	    sscanf(segments[i].begin,"%d:%lf", &fpair.featId, &fpair.val);
	    vec_feat.push_back(fpair);
	}
}

void item::init_dynamicSim_feat_vec()
{
    int n = 1; //this might change, since we might append, starts with 1:sumSim; 
    for (int i = 0; i < n; i++)
	{
	    feat_pair tmp;
	    tmp.featId = i+1;
	    tmp.val = 0;
	    vec_dynamicSim_feat_.push_back(tmp);
	}
}

void item::parse_vw_input(char *line)
{
    parse_label(line);
    //  strcpy(expanded_line_, line);
    //  substring stmp = {expanded_line_, expanded_line_ + strlen(line)};
     substring stmp = {line, line + strlen(line)}; 
    // after first bar follows the quality features
    char * first_barloc = safe_index(stmp.begin, '|', stmp.end);
    // after second bar follows the similiary features
    char * second_barloc = safe_index(first_barloc + 1, '|', stmp.end);
    // construct vec_feat for quality features
    construct_feat_vec(first_barloc+1, second_barloc-1, vec_quality_feat_);
    // construct vec_feat for user-specified similiarty features, but they will not be used for regression/ranking
    construct_feat_vec(second_barloc+1, stmp.end, vec_sim_feat_);
    /*
    int numcopychars = second_barloc - line;
    //  strncpy( expanded_line_, line, numcopychars);
    // expanded_line_[numcopychars] = '\0';  //have to manually put end 
    strncpy( prefix_line_, line, numcopychars);
    prefix_line_[numcopychars] = '\0';  //have to manually put end 
    */
    
    int numcopychars = second_barloc - first_barloc;
    strncpy( prefix_line_, first_barloc, numcopychars);
    prefix_line_[numcopychars] = '\0';  //have to manually put end 
}

void item::construct_label_line()
{
    sprintf(label_line_, "%lf %lf %d", label_, importance_, envId_); 
}

void item::construct_expanded_line()  //should be called after init_dynamicSim_feat_vec
{
    construct_label_line();
    strcpy(expanded_line_, label_line_);
    strcat(expanded_line_, prefix_line_);
    // strcpy(expanded_line_, prefix_line_);
    char append_line[charN] = " |dsf";
    /*
    for (int i = 0; i < vec_dynamicSim_feat_.size(); i++)
	{
	    char tmpPair[50];
	    sprintf(tmpPair, " %d:%.6lf", vec_dynamicSim_feat_[i].featId, vec_dynamicSim_feat_[i].val);
	    strcat(append_line, tmpPair);
	}
    */
    strcat(append_line, construct_feat_line(vec_dynamicSim_feat_));
    strcat(expanded_line_, append_line);
    //  cout<<expanded_line_<<endl;
}

char * item::construct_feat_line(vector<feat_pair> & vec_feat)
{
    char feat_line[charN] = "\0";
    // char* feat_line = new char[charN];
    for (int i = 0; i < vec_feat.size(); i++)
	{
	    char tmpPair[50];
	    sprintf(tmpPair, " %d:%.6lf", vec_feat[i].featId, vec_feat[i].val);
	    strcat(feat_line, tmpPair);
	    //	    cout << feat_line << endl;
	}
    return feat_line;
}

void item::init_example(vw* model,  char * line)
{
    parse_vw_input(line);
    init_dynamicSim_feat_vec();
    construct_expanded_line();
    //  ex = VW::read_example(*model, expanded_line_);
}

void item::update_example(vw* model, vector<feat_pair> & vecFeat2, double newLabel, bool updateLabel ) //
{
    update_dynamicSimFeature(vecFeat2);  //update vec_dynamicsSim_feat_
    //    cout << "##" <<endl;
    if (updateLabel)
	label_ = newLabel;
    construct_expanded_line();
    //    VW::finish_example(*model, ex);
    // release_vwexample(model);
    //    ex = VW::read_example(*model, expanded_line_);     
}

vector<feat_pair>  item::get_diff_vec_feat(vector<feat_pair> & vec1, vector<feat_pair> & vec2 )
{
    vector<feat_pair> vec;
    if (vec1.size() != vec2.size())
	cerr << "mismatch of item pairwise difference " <<endl;
    for (int i = 0; i < vec1.size(); i++)
	{
	    if (vec1[i].featId != vec2[i].featId)
		{
		    cerr << "mismatch of featId for item pairwise difference" <<endl;
		}
	    feat_pair tmp;
	    tmp.featId = vec1[i].featId;
	    tmp.val = vec1[i].val - vec2[i].val;
	    vec.push_back(tmp);
	}
    return vec;
}
void item::get_diff_str(item & itm, char * diff_line)
{
    double diff_label, diff_importance;
    // char diff_line[charN];
    //forming binary label
    if (label_ < itm.get_label())
	{
	    diff_label = -1;
	}
    else if (label_ > itm.get_label())
	{
	    diff_label = 1;
	}
    else  //equal
	{
	    strcpy(diff_line, "\0");
	    return;
	}
    //get importance
    diff_importance = fabs(label_ - itm.get_label());
    //form label part
    sprintf(diff_line, "%lf %lf %d", diff_label, diff_importance, envId_); 
    //   cout << diff_line <<endl;
    //form quality feature part
    strcat(diff_line, "|qf");
    vector<feat_pair> vec_diff_qf= get_diff_vec_feat(vec_quality_feat_, itm.get_vec_quality_feat());
    /*
    for (int i = 0; i < tmp.size(); i++)
	{
	    cout << tmp[i].featId <<",,,," << tmp[i].val << endl;
	}
    */
    // strcat(diff_line, construct_feat_line(get_diff_vec_feat(vec_quality_feat_, itm.get_vec_quality_feat())) );
    strcat(diff_line, construct_feat_line(vec_diff_qf) );

    // cout << diff_line <<endl;
    strcat(diff_line, " |dsf");
    vector<feat_pair> vec_diff_dsf = get_diff_vec_feat(vec_dynamicSim_feat_, itm.get_vec_dynamicSim_feat());
    //  strcat(diff_line,  construct_feat_line(get_diff_vec_feat(vec_dynamicSim_feat_, itm.get_vec_dynamicSim_feat()) ) );
    strcat(diff_line,  construct_feat_line(vec_diff_dsf ) );
    //cout << "diffline " << diff_line <<endl;
    
    //  return diff_line;
}

//0 means completely similar = 0 distance
double item::compute_simDis (item & itm)
{
    //use the inner product of vec_sim_feat_
    //need to normalize the vectors
    vector<feat_pair> vecFeat1 = get_vec_sim_feat();
    vector<feat_pair> vecFeat2 = itm.get_vec_sim_feat(); 
    int ind1, ind2;
    ind1 = ind2 = 0;
    double norm1, norm2;
    norm1 = norm2 = 0;
    double dotprod = 0;
    while (ind1 < vecFeat1.size() && ind2 < vecFeat2.size())
    {
	if (vecFeat1[ind1].featId < vecFeat2[ind2].featId)
	    ind1++;
	else if (vecFeat1[ind1].featId > vecFeat2[ind2].featId)
	    ind2++;
	else 
	    {
		dotprod += vecFeat1[ind1].val * vecFeat2[ind2].val;
		ind1++;
		ind2++;
	    }
    }
    //compute norm
    for (int i = 0; i < vecFeat1.size(); i++)
    {
	norm1 += vecFeat1[i].val * vecFeat1[i].val;  
    }
    for (int i = 0; i < vecFeat2.size(); i++)
    {
	norm2 += vecFeat2[i].val * vecFeat2[i].val;  
    }
    return 1 - dotprod/sqrt(norm1*norm2);  //in the range of [0, 2]
   
}

void item::update_dynamicSimFeature(vector<feat_pair> & vecFeat2)
{
    //  cout << "update_dynamicsSimFeature" <<endl;
    // vector<feat_pair> vecFeat1 = get_vec_dynamicSim_feat();   
    int ind1, ind2;
    ind1 = ind2 = 0;
    while (ind1 < vec_dynamicSim_feat_.size() && ind2 < vecFeat2.size())
	{
	    if (vec_dynamicSim_feat_[ind1].featId < vecFeat2[ind2].featId)
		ind1++;
	    else if (vec_dynamicSim_feat_[ind1].featId > vecFeat2[ind2].featId)
		ind2++;
	    else 
		{
		    vec_dynamicSim_feat_[ind1].val = vecFeat2[ind2].val;
		    ind1++;
		    ind2++;
		
		}
	}
}























