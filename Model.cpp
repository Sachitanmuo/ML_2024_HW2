#include "Model.h"

Model::Model(){
    read_file();
}

void Model::read_file(){
    vector<vector<double>> team(4, vector<double>(4));
    team[0] = {1, 0, 0, 0};
    team[1] = {0, 1, 0, 0};
    team[2] = {0, 0, 1, 0};
    team[3] = {0, 0, 0, 1};

    // Training file
    ifstream file(training_file);
    string line;
    getline(file, line); // Skip the header line
    while (getline(file, line)) {
        stringstream ss(line);
        string team_str, offensive_str, defensive_str;
        getline(ss, team_str, ',');
        getline(ss, offensive_str, ',');
        getline(ss, defensive_str, ',');

        Data data;
        data.team = stoi(team_str);
        data.team_vector = team[data.team];
        data.Offensive = stod(offensive_str);
        data.Defensive = stod(defensive_str);
        Training_set.push_back(data);
    }

    // Testing file
    ifstream file_(testing_file);
    getline(file_, line); // Skip the header line
    while (getline(file_, line)) {
        stringstream ss(line);
        string team_str, offensive_str, defensive_str;
        getline(ss, team_str, ',');
        getline(ss, offensive_str, ',');
        getline(ss, defensive_str, ',');

        Data data;
        data.team = stoi(team_str);
        data.team_vector = team[data.team];
        data.Offensive = stod(offensive_str);
        data.Defensive = stod(defensive_str);
        Testing_set.push_back(data); // Store data in Testing_set
    }

    //generate the 3-class data
    vector<vector<double>> team_(3, vector<double>(3));
    team_[0] = {1, 0, 0};
    team_[1] = {0, 1, 0};
    team_[2] = {0, 0, 1};

    for(auto& data : Training_set){
        Data d;
        d.Offensive = data.Offensive;
        d.Defensive = data.Defensive;
        switch (data.team){
            case 0:
                d.team = 0;
                d.team_vector = team_[0];
                break;
            case 1:
                d.team = 1;
                d.team_vector = team_[1];
                break;
            case 2:
                d.team = 2;
                d.team_vector = team_[2];
                break;
            case 3:
                d.team = 0;
                d.team_vector = team_[0];
                break;
        }
        Training_set_p2.push_back(d);
    }

    for(auto& data : Testing_set){
        Data d;
        d.Offensive = data.Offensive;
        d.Defensive = data.Defensive;
        switch (data.team){
            case 0:
                d.team = 0;
                d.team_vector = team_[0];
                break;
            case 1:
                d.team = 1;
                d.team_vector = team_[1];
                break;
            case 2:
                d.team = 2;
                d.team_vector = team_[2];
                break;
            case 3:
                d.team = 0;
                d.team_vector = team_[0];
                break;
        }
        Testing_set_p2.push_back(d);
    }
}

void Model::Train_Generative(){
    Training_mean = calculate_mean(Training_set);
    covariance_matrix = calculate_cov_mat(Training_set);
    Train_prob = calculate_prob(Training_set);
    vector<Eigen::VectorXd> u(4, Eigen::VectorXd(2));
    Eigen::VectorXd X(2);
    for(int i = 0; i < 4; i++){
        u[i] << Training_mean[i].Offensive, Training_mean[i].Defensive;
        cout << u[i] << " " << endl;
    }
    
    for(int i = 0; i < 4; i++){
        W[i] = covariance_matrix.inverse() * u[i];
        cout << "W["<<i<<"]: \n"<< W[i] << endl;
        W0[i] = (-0.5) * u[i].transpose() * covariance_matrix.inverse() * u[i] * log(Train_prob[i]);
    }
    vector<Eigen::VectorXd> y_pred(Training_set.size(), Eigen::VectorXd(4));
    vector<Eigen::VectorXd> ground_truth(Training_set.size(), Eigen::VectorXd(4));

    for(int i = 0; i < Training_set.size(); i++){
        vector<double> d(2);
        d[0] = Training_set[i].Offensive;
        d[1] = Training_set[i].Defensive;
        Eigen::VectorXd y_pred_vec(4);
        Eigen::VectorXd ground_truth_vec(4);
        vector<double> y_pred_temp = predict_generative(d);
        vector<double> ground_truth_temp = Training_set[i].team_vector;
        for(int j = 0; j < 4; j++){
            y_pred_vec[j] = y_pred_temp[j];
            ground_truth_vec[j] = ground_truth_temp[j];
        }
        y_pred[i] = y_pred_vec;
        ground_truth[i] = ground_truth_vec;

    }

    Eigen::MatrixXd Conf_Mtx(Training_set[0].team_vector.size(), Training_set[0].team_vector.size());
    Conf_Mtx = cal_confusion_matrix(y_pred, ground_truth);
    cout << "========== Confusion Matrix ============" << endl;
    int max_width = 0;
    for(int i = 0; i < Conf_Mtx.rows(); i++){
        for(int j = 0; j < Conf_Mtx.cols(); j++){
            stringstream ss;
            ss << Conf_Mtx(i, j);
            int element_width = ss.str().length();
            max_width = max(max_width, element_width);
        }
    }
    for(int i = 0; i < Conf_Mtx.rows(); i++){
        for(int j = 0; j < Conf_Mtx.cols(); j++){
            cout << setw(max_width + 1) << Conf_Mtx(i, j) << " ";
        }
        cout << endl;
    }
    cout << "=========================================" << endl;

    //output weights
    ofstream out("generative_weights.txt");
    for(int i = 0; i < 4; i++){
        out << W[i](0) << " " <<W[i](1) << " " << W0[i] << endl;
    }

}

void Model::Train_Discriminative(){
    //initialize weights
    vector<Eigen::VectorXd> weights(4, Eigen::VectorXd(3));
    vector<Eigen::VectorXd> delta_W(4, Eigen::VectorXd(3));
    vector<double> a_k(4);
    Eigen::VectorXd y(4); //predicted
    Eigen::VectorXd t(4); // True Label
    vector<Eigen::VectorXd> phi(Training_set.size(), Eigen::VectorXd(3));
    for(int i = 0; i < Training_set.size(); i ++){
        phi[i](0) = 1;
        phi[i](1) = Training_set[i].Offensive;
        phi[i](2) = Training_set[i].Defensive;
    }
    for(int i = 0; i < weights.size() ; i++){
        for(int j = 0; j < 3; j++){
            weights[i] << 1, 1, 1;
        }
    }
    //Training parameters
    double total = 0;
    int epochs = 100000;
    double learning_rate = 1e-6;
    for(int i = 0; i < epochs ; i++){
        for(int x = 0; x < 4; x++){
            delta_W[x] << 0, 0, 0;
        }
        for(auto j = 0 ; j < Training_set.size();j++){
            total = 0;
            for(int k = 0; k < 4; k++){
                a_k[k] = exp(weights[k].transpose() * phi[j]);
                total += a_k[k];
            }
            
            t << Training_set[j].team_vector[0], 
                     Training_set[j].team_vector[1], 
                     Training_set[j].team_vector[2], 
                     Training_set[j].team_vector[3];
            for(int k = 0; k < 4; k++){
                y[k] = a_k[k] / total;
                delta_W[k] += (y(k) - t(k)) * phi[j];
            }
        }
        cout << "epoch " << i + 1 << " : " << endl;
        cout << "predicted:\n" << y << "\ntrue: \n" << t << endl;
        cout << "\ndelta_W: \n" << delta_W[0] << endl;
        //getchar();
        //update the weights
        for(int x = 0; x < 4; x++){
            weights[x] -= learning_rate * delta_W[x];
        }
    }  
    //finish training
    weights_dis = weights;

    vector<Eigen::VectorXd> y_pred(Training_set.size(), Eigen::VectorXd(4));
    vector<Eigen::VectorXd> ground_truth(Training_set.size(), Eigen::VectorXd(4));
    for(int i = 0; i < Training_set.size(); i++){
        vector<double> d(2);
        d[0] = Training_set[i].Offensive;
        d[1] = Training_set[i].Defensive;
        Eigen::VectorXd y_pred_vec(4);
        Eigen::VectorXd ground_truth_vec(4);
        vector<double> y_pred_temp = predict_discriminative(d);
        vector<double> ground_truth_temp = Training_set[i].team_vector;
        for(int j = 0; j < 4; j++){
            y_pred_vec[j] = y_pred_temp[j];
            ground_truth_vec[j] = ground_truth_temp[j];
        }
        y_pred[i] = y_pred_vec;
        ground_truth[i] = ground_truth_vec;
    }
    Eigen::MatrixXd Conf_Mtx(Training_set[0].team_vector.size(), Training_set[0].team_vector.size());
    Conf_Mtx = cal_confusion_matrix(y_pred, ground_truth);
    cout << "==========Confusion Matrix ============" << endl;
    int max_width = 0;
    for(int i = 0; i < Conf_Mtx.rows(); i++){
        for(int j = 0; j < Conf_Mtx.cols(); j++){
            stringstream ss;
            ss << Conf_Mtx(i, j);
            int element_width = ss.str().length();
            max_width = max(max_width, element_width);
        }
    }
    for(int i = 0; i < Conf_Mtx.rows(); i++){
        for(int j = 0; j < Conf_Mtx.cols(); j++){
            cout << setw(max_width + 1) << Conf_Mtx(i, j) << " ";
        }
        cout << endl;
    }
    cout << "=======================================" << endl;

    //calculate the accuracy
    double accuracy = cal_acc(Conf_Mtx);
    cout << "Accuracy: " << accuracy << endl;
    //========output weights==================
    ofstream out("discriminative_weights.txt");
    for(auto& w : weights_dis){
        for(int i = 0; i < w.size() ; i++){
            out << w(i) << " ";
        }
        out << endl;
    }
}
vector<double> Model::calculate_prob(vector<Data>& data){
    int N = data.size();
    vector<double> prob(4, 0);
    for(auto d: data){
        prob[d.team]++;
    }

    return prob;
}

vector<Mean> Model::calculate_mean(vector<Data>& data){
    int N = data.size();
    vector<Mean> m(4);
    int count[4] = {0};
    for(auto& i:m){
        i.Offensive = 0;
        i.Defensive = 0;
    }
    for(auto& i: data){
        count[i.team]++;
        m[i.team].Offensive += i.Offensive;
        m[i.team].Defensive += i.Defensive;
        mean_[0] += i.Offensive;
        mean_[1] += i.Defensive;
    }
    int count_ = 0;
    for(int i = 0; i < 4; i++){
        m[i].Offensive /= count[i];
        m[i].Defensive /= count[i];
        count_ += count[i];
    }
    mean_[0] /= count_;
    mean_[1] /= count_;
    return m;
}

void Model::calculate_cov_mat_(vector<Data>& data){
    for(int k = 0; k < data[0].team_vector.size(); k ++){
        double mean_Offensive = mean_[0];
        double mean_Defensive = mean_[1];
        Eigen::MatrixXd cov_mat(2, 2);
        int count = 0;
        for(auto& d: data){
                count++;
                Eigen::VectorXd vec(2);
                vec << d.Offensive - mean_Offensive, d.Defensive - mean_Defensive;
                cov_mat += vec * vec.transpose();
        }
        cov_mat /= (count - 1);
        for(int i = 0; i < cov_mat.rows() ; i ++){
            for(int j = 0; j < cov_mat.cols() ; j++){
                cout << cov_mat(i, j) << " ";
            }
            cout << endl;
        }
        CM[k] = cov_mat;
    }
    
}

Eigen::MatrixXd Model::calculate_cov_mat(vector<Data>& data){
        Eigen::MatrixXd cov_mat(2, 2);
        int count = 0;
        for(auto& d: data){
                count++;
                Eigen::VectorXd vec(2);
                vec << d.Offensive - Training_mean[d.team].Offensive, d.Defensive - Training_mean[d.team].Defensive;
                cov_mat += vec * vec.transpose();
                //cout << vec << endl;
        }
        cov_mat /= (count - 1);
        for(int i = 0; i < cov_mat.rows() ; i ++){
            for(int j = 0; j < cov_mat.cols() ; j++){
                cout << cov_mat(i, j) << " ";
            }
            cout << endl;
        }
        return cov_mat;
    
}
vector<double> Model::predict_generative(vector<double>& data){
    vector<double> y(4, 0);
    double total = 0;
    //softmax
    for(int i = 0; i < 4; i++){
        y[i] = exp(W[i](0) * data[0] + W[i](1) * data[1] + W0[i](0, 0));
        total += y[i];
    }
    for(auto& i : y){
        i /= total;
    }
    return y;
}

vector<double> Model::predict_discriminative(vector<double>& data){
    vector<double> y(4, 0);
    Eigen::VectorXd x(3);
    x << 1, data[0], data[1];
    double total = 0;
    //softmax
    for(int i = 0; i < 4; i++){
        y[i] = exp(weights_dis[i].transpose() * x);
        total += y[i];
    }

    return y;
}
Eigen::MatrixXd Model::cal_confusion_matrix(vector<Eigen::VectorXd> y_pred, vector<Eigen::VectorXd> t){
    int size = y_pred[0].size();
    Eigen::MatrixXd confusion_matrix = Eigen::MatrixXd::Zero(size, size);
    for(int i = 0; i < y_pred.size(); i++){
        int pred_index, true_index;
        y_pred[i].maxCoeff(&pred_index);
        t[i].maxCoeff(&true_index);
        confusion_matrix(pred_index, true_index) += 1;
    }
    return confusion_matrix;
}

double Model::cal_acc(Eigen::MatrixXd & cm){
    double diagonal_sum = 0;
    double total_sum = 0;
    for(int i = 0; i < cm.rows(); i++){
        for(int j = 0; j < cm.cols(); j++){
            total_sum += cm(i, j);
            if(i == j){
                diagonal_sum += cm(i, j);
            }
        }
    }
    return diagonal_sum / total_sum;
}

//double Model::cal_gaussian_prob(Eigen::VectorXd &x, GaussianDistribution &g){
    
//}