/*************************************************
Author: Tianwen Jiang
Date: 2016-01-27
Description: the implementation of neural network
**************************************************/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "tw_func.h"
#include "neural_network2.h"
#include "load_handwrite_data.h"

using namespace std;

/*************************************************
Function: main()
Description: main control program
Input:
    int argc: the number of the arguments
    char **argv: the arguments
Output: the setting of model
Return:
    the state of excution
Others:
*************************************************/
int main(int argc, char **argv)
{
    vector<int> nn_size;
    int index, layers_num;
    vector<vector<double> > training_inputs;
    vector<vector<double> > training_outputs;
    vector<vector<double> > test_inputs;
    vector<vector<double> > test_outputs;
    int nepochs = 30;
    int batch_size = 10;
    double rate = 3.0;
    string cost_method = "CEC";

    if ((index = have_arg((char *)"-layers_num", argc, argv)) > 0) layers_num = atoi(argv[index+1]);
    else
    {
        cout << "please set the value of -layers_num" << endl;
        exit(1);
    }
    if ((index = have_arg((char *)"-size", argc, argv)) > 0)
    {
        for (int i = 1; i <= layers_num; i++)
            nn_size.push_back(atoi(argv[index+i]));
    }
    else
    {
        cout << "please set the value of -size" << endl;
        exit(1);
    }

    if ((index = have_arg((char *)"-nepochs", argc, argv)) > 0) nepochs = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-batch_size", argc, argv)) > 0) batch_size = atoi(argv[index+1]);
    if ((index = have_arg((char *)"-rate", argc, argv)) > 0) rate = atof(argv[index+1]);
    if ((index = have_arg((char *)"-cost_method", argc, argv)) > 0) cost_method = argv[index+1];

    cout << "layers num = " << layers_num << endl;
    cout << "size = ";
    for (unsigned int i = 0; i < nn_size.size(); i++)
        cout << nn_size[i] << " ";
    cout << endl;
    cout << "nepochs = " << nepochs << endl;
    cout << "batch_size = " << batch_size << endl;
    cout << "learning rate = " << rate << endl;
    cout << "cost_method = " << cost_method << endl;

    load_training_data(training_inputs, training_outputs);
    load_testing_data(test_inputs, test_outputs);

    if (cost_method == "CEC")
    {
        Neural_Network<CrossEntropyCost> neural_network(nn_size);
        neural_network.SGD(training_inputs, training_outputs, nepochs,
                        batch_size, rate, test_inputs, test_outputs);
    }
    else if (cost_method == "QC")
    {
        Neural_Network<QuadraticCost> neural_network(nn_size);
        neural_network.SGD(training_inputs, training_outputs, nepochs,
                        batch_size, rate, test_inputs, test_outputs);
    }
    else
    {
        cout << "no such cost method" << endl;
        exit(1);
    }

    return 0;
}
