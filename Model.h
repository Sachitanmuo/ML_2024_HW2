#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<cmath>
#include "eigen-3.3.7/Eigen/Eigen"
#include <iomanip>
using namespace std;
struct Data{
    double Offensive;
    double  Defensive;
    int team;
    vector<double> team_vector;
};

struct Mean{
    double Offensive;
    double Defensive;
};

struct GaussianDistribution{
    vector<double> mean_vector;
    vector<vector<double>> covariance_matrix;
};


class Model{
private:
    vector<Data> Training_set;
    vector<Data> Testing_set;
    vector<Data> Training_set_p2;
    vector<Data> Testing_set_p2;
    string training_file = "HW2_training.csv";
    string testing_file = "HW2_testing.csv";
    vector<double> Train_prob;
    vector<Mean> Training_mean;
    vector<Mean> Testing_Mean;
    double mean_[2];
    vector<GaussianDistribution> gaussian_distributions;
    Eigen::MatrixXd covariance_matrix;
    Eigen::MatrixXd CM[4]; //4 covarience matrix
    Eigen::MatrixXd W[4]; //Generative
    double W0[4]; // Generative
    Eigen::MatrixXd W_[4]; //Generative part 2
    double W0_[4]; // Generative part 2
    vector<Eigen::VectorXd> weights_dis; //Discriminative
    vector<Eigen::VectorXd> weights_dis_p2; // part 2
    vector<Eigen::VectorXd> a;
    int epochs;
public:
    Model(int epochs): epochs(epochs) {read_file();}
    void read_file();
    void Train_Generative();
    void Train_Generative_part2();
    vector<double> calculate_prob(vector<Data>&);
    vector<Mean> calculate_mean(vector<Data>&);
    Eigen::MatrixXd calculate_cov_mat(vector<Data>&);
    void calculate_cov_mat_(vector<Data>&);
    void Train_Discriminative();
    void Train_Discriminative_part2();
    vector<double> predict_generative(vector<double>&); // predict which team
    vector<double> predict_generative_p2(vector<double>&);
    //Eigen::VectorXd logit_func(Eigen::VectorXd&, Eigen::VectorXd&);
    vector<double> predict_discriminative(vector<double>&);
    vector<double> predict_discriminative_p2(vector<double>&);
    Eigen::MatrixXd cal_confusion_matrix(vector<Eigen::VectorXd> y_pred, vector<Eigen::VectorXd> t);
    double cal_acc(Eigen::MatrixXd &);
    //double cal_gaussian_prob(Eigen::VectorXd&, GaussianDistribution&);
    void Test_Generative();
    void Test_Discriminative();
    void Test_Discriminative_part2();
    void Test_Generative_part2();
};