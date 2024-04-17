#include<iostream>
#include "Model.h"
using namespace std;
int main(int argc, char* argv[]){
    Model model(100000);
    model.Train_Generative();
    model.Test_Generative();
    model.Train_Discriminative();
    model.Test_Discriminative();

    //part 2
    model.Train_Generative_part2();
    model.Test_Generative_part2();
    model.Train_Discriminative_part2();
    model.Test_Discriminative_part2();
}