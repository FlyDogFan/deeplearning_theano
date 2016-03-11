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

using namespace std;

/*************************************************
Function: sigmoid()
Description: get the sigmoid value for given z
Input:
    double z: value
Output:
Return: sigmoid value for given z
Others:
*************************************************/
double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

/*************************************************
Function: rand_max()
Description: produce the random value in [0,x)
Input: max value x
Output:
Return: the random value in [0,x)
Others:
*************************************************/
int rand_max(int x)
{
    int res = (rand()*rand())%x;
    while (res<0)
        res+=x;
    return res;
}

/*************************************************
Function: gaussrand()
Description: produce gaussrand, expectation=0, variance=1
Input:
Output:
Return:
Others: x = x * variance + expectation
*************************************************/
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

double sqr(double x)
{
    return x*x;
}

class Neural_Network
{
private:
    vector<int> nn_size;
    int layers_num;
    vector<vector<double> > biases;
    vector<vector<vector<double> > > weights;
    vector<vector<double> > nabla_biases;
    vector<vector<vector<double> > > nabla_weights;
    vector<vector<double> > delta_nabla_b;
    vector<vector<vector<double> > > delta_nabla_w;

    vector<double> cost_derivative(vector<double> &output_activation, double &y)
    {
        vector<double> result;
        for (unsigned int i = 0; i < output_activation.size(); i++)
        {
            result.push_back(output_activation[i]);
        }
        result[(int)y] = result[(int)y]-1;
        return result;
    }

    double calc_loss(vector<double> &output1, vector<double> &output2)
    {
        double sqr_sum = 0;
        if (output1.size() != output2.size())
        {
            cout << "wrong!" << endl;
            exit(1);
        }
        for (unsigned i = 0; i < output1.size(); i++)
        {
            sqr_sum += sqr(output1[i]-output2[i]);
        }
        return sqr_sum / 2;
    }

    /** back propagation algorithm **/
    void backprop(vector<double> &x, double &y)
    {
        for (unsigned int i = 0; i < delta_nabla_b.size(); i++)
            for (unsigned int j = 0; j < delta_nabla_b[i].size(); j++)
                delta_nabla_b[i][j] = 0;

        for (unsigned int i = 0; i < delta_nabla_w.size(); i++)
            for (unsigned int j = 0; j < delta_nabla_w[i].size(); j++)
                for (unsigned int k = 0; k < delta_nabla_w[i][j].size(); k++)
                    delta_nabla_w[i][j][k] = 0;

        vector<double> activation;
        vector<double> z;
        vector<vector<double> > activations;
        vector<vector<double> > zs;

        // Input x: set the corresponding activation a^1 for input layer
        //activation = new vector<double>();
        for (unsigned int i = 0; i < x.size(); i++)
            activation.push_back(x[i]);
        activations.push_back(activation);

        // FeedForward: for each l=1,2, compute z^l=w^la^{l-1}+b^l and a^l=\sigma(z^l)
        // current level
        for (unsigned int i = 1; i < nn_size.size(); i++)
        {
            activation.clear();
            z.clear();
            // every neural in the current level
            for (int j = 0; j < nn_size[i]; j++)
            {
                double w_dot_a = 0;

                // every neural in the last level
                for (int k = 0; k < nn_size[i-1]; k++)
                    w_dot_a += weights[i-1][j][k] * (activations[i-1][k]);
                z.push_back(w_dot_a + biases[i-1][j]);
                activation.push_back(sigmoid(z.back()));

            }
            zs.push_back(z);
            activations.push_back(activation);
        }

        // output error delta^L: compute the vector delta^L=nabla_a C odot sigma'(z^L)
        vector<double> delta;
        vector<double> delta_tmp = cost_derivative(activations.back(), y);
        vector<double> z_L = zs.back();
        for (unsigned int i = 0; i < z_L.size(); i++)
        {
            delta.push_back(delta_tmp[i] * sigmoid_prime(z_L[i]));
        }

        // backpropagate the error;
        for (int j = 0; j < nn_size[layers_num-1]; j++)
        {
            (delta_nabla_b.back())[j] = delta[j];
            for (int k = 0; k < nn_size[layers_num-2]; k++)
                (delta_nabla_w.back())[j][k] = activations[layers_num-2][k] * delta[j];
        }
        for (int layer = layers_num-1; layer >= 2; layer--)
        {
            vector<double> old_delta = delta;
            delta.clear();
            for (int j = 0; j < nn_size[layer-1]; j++)
            {
                double delta_element = 0;
                for (int k = 0; k < nn_size[layer]; k++)
                    delta_element += weights[layer-1][k][j]*old_delta[k];
                delta_element = delta_element * sigmoid_prime(zs[layer-2][j]);
                delta.push_back(delta_element);

                //update delta_nabla_w and delta_nabla_b
                delta_nabla_b[layer-2][j] = delta[j];
                for (int k = 0; k < nn_size[layer-2]; k++)
                    delta_nabla_w[layer-2][j][k] = activations[layer-2][k] * delta[j];
            }
        }
    }

    /** use the batch_data to update weights and biases **/
    void update_weights_biases(vector<vector<double> > &batch_inputs,
                                        vector<double> &batch_outputs, double rate)
    {
        //cout << "in update_weights_biases" << endl;
        int batch_len = batch_inputs.size();

        for (unsigned int i = 0; i < nabla_biases.size(); i++)
            for (unsigned int j = 0; j < nabla_biases[i].size(); j++)
                nabla_biases[i][j] = 0;

        for (unsigned int i = 0; i < nabla_weights.size(); i++)
            for (unsigned int j = 0; j < nabla_weights[i].size(); j++)
                for (unsigned int k = 0; k < nabla_weights[i][j].size(); k++)
                    nabla_weights[i][j][k] = 0;

        for (int index = 0; index < batch_len; index++)
        {
            backprop(batch_inputs[index], batch_outputs[index]);

            for (unsigned int i = 0; i < nabla_biases.size(); i++)
                for (unsigned int j = 0; j < nabla_biases[i].size(); j++)
                    nabla_biases[i][j] += delta_nabla_b[i][j];

            for (unsigned int i = 0; i < nabla_weights.size(); i++)
                for (unsigned int j = 0; j < nabla_weights[i].size(); j++)
                    for (unsigned int k = 0; k < nabla_weights[i][j].size(); k++)
                        nabla_weights[i][j][k] += delta_nabla_w[i][j][k];
        }

        for (unsigned int i = 0; i < biases.size(); i++)
            for (unsigned int j = 0; j < biases[i].size(); j++)
            {
                biases[i][j] = biases[i][j]-rate*nabla_biases[i][j]/batch_len;
                //cout << rate*nabla_biases[i][j]/batch_len <<endl;
            }

        for (unsigned int i = 0; i < weights.size(); i++)
            for (unsigned int j = 0; j < weights[i].size(); j++)
                for (unsigned int k = 0; k < weights[i][j].size(); k++)
                    weights[i][j][k] = weights[i][j][k]-rate*nabla_weights[i][j][k]/batch_len;
    }

    vector<double> feedforword(vector<double> &input)
    {
        vector<double> activation;

        for (unsigned int i = 0; i < input.size(); i++)
            activation.push_back(input[i]);

        for (unsigned int i = 1; i < nn_size.size(); i++)
        {
            vector<double> old_activation = activation;
            activation.clear();
            /* every neural in the current level */
            for (int j = 0; j < nn_size[i]; j++)
            {
                double w_dot_a = 0;

                /* every neural in the last level */
                for (int k = 0; k < nn_size[i-1]; k++)
                    w_dot_a += weights[i-1][j][k] * (old_activation[k]);
                activation.push_back(sigmoid(w_dot_a + biases[i-1][j]));
            }
        }
        return activation;
    }

    int evaluate(vector<vector<double> > &testing_inputs, vector<double> &testing_outputs)
    {
        int right_count = 0;
        vector<double> output;
        for (unsigned int i = 0; i < testing_inputs.size(); i++)
        {
            output = feedforword(testing_inputs[i]);
            int max_pos = 0;
            double max_ele = 0;
            for (unsigned int j = 0; j < output.size(); j++)
            {
                if (output[j] > max_ele)
                {
                    max_ele = output[j];
                    max_pos = j;
                }
            }
            if (max_pos == (int)testing_outputs[i])
                right_count++;
        }
        return right_count;
    }

public:
    /** construction fuction: initialize the weights and biases **/
    Neural_Network(vector<int> &nn_size_in)
    {
        layers_num = nn_size_in.size();
        nn_size = nn_size_in;

        cout << endl << "initializing the biases and weights ... ..." << endl;

        /** initializing the biases **/
        biases.resize(layers_num-1);
        nabla_biases.resize(layers_num-1);
        delta_nabla_b.resize(layers_num-1);
        for (unsigned int i = 0; i < biases.size(); i++)
        {
            biases[i].resize(nn_size[i+1]);
            nabla_biases[i].resize(nn_size[i+1]);
            delta_nabla_b[i].resize(nn_size[i+1]);
        }
        for (unsigned int i = 0; i < biases.size(); i++)
            for (unsigned int j = 0; j < biases[i].size(); j++)
                biases[i][j] = gaussrand();

        /** initializing the weights **/
        weights.resize(layers_num-1);
        nabla_weights.resize(layers_num-1);
        delta_nabla_w.resize(layers_num-1);
        for (unsigned int i = 0; i < weights.size(); i++)
        {
            weights[i].resize(nn_size[i+1]);
            nabla_weights[i].resize(nn_size[i+1]);
            delta_nabla_w[i].resize(nn_size[i+1]);
            for (unsigned int j = 0; j < weights[i].size(); j++)
            {
                weights[i][j].resize(nn_size[i]);
                nabla_weights[i][j].resize(nn_size[i]);
                delta_nabla_w[i][j].resize(nn_size[i]);
            }
        }
        for (unsigned int i = 0; i < weights.size(); i++)
            for (unsigned int j = 0; j < weights[i].size(); j++)
                for (unsigned int k = 0; k < weights[i][j].size(); k++)
                    weights[i][j][k] = gaussrand();

        cout << "initialized ok. " << endl << endl;
    }

    /** train the neural network using stochastic gradient descent **/
    void SGD(vector<vector<double> > &training_inputs,
             vector<double> &training_outputs, int nepochs, int batch_size, double rate,
             vector<vector<double> > &testing_inputs, vector<double> &testing_outputs)
    {
        //cout << "in SGD." << endl;
        int training_len = training_inputs.size();
        int nbatchs = training_len/batch_size;

        vector<vector<double> > batch_inputs;
        vector<double> batch_outputs;
        batch_inputs.resize(batch_size);
        batch_outputs.resize(batch_size);

        for (int epoch = 0; epoch < nepochs ; epoch++)
        {
            for (int batch = 0; batch < nbatchs; batch++)
            {
                /** for each batch we select batch_size training data for training model **/
                for (int i = 0; i < batch_size; i++)
                {
                    int index = rand_max(training_len);
                    batch_inputs[i] = training_inputs[index];
                    batch_outputs[i] = training_outputs[index];
                }

                /** use the batch_data to update weights and biases **/
                update_weights_biases(batch_inputs, batch_outputs, rate);
            }
            //int right_count = 0;
            int right_count = evaluate(testing_inputs, testing_outputs);
            cout << "epoch: " << epoch << "\t" << right_count << "/" << testing_inputs.size() << endl;
            //loss = loss / training_len;
            //cout << "epoch: " << epoch << "\t" << "loss: " << loss << endl;
        }
    }
};

/*************************************************
Function: load_training_data()
Description: load training data from
Input:
    vector<vector<int> > training_inputs
    vector<int> training_outputs
Output: the info of training data
Return:
Others: update the training_inputs and training_outputs
*************************************************/
void load_training_data(vector<vector<double> > &training_inputs, vector<double> &training_outputs)
{
    char ch;
    int data_size = 4, magic_num = 0, image_num = 0, rows_num = 0, cols_num = 0, pixel, label;
    FILE *training_image_file = fopen("./data/train-images.idx3-ubyte", "rb");
    cout << endl << "loading the train images ... ..." << endl;
    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_image_file);
        magic_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_image_file);
        image_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_image_file);
        rows_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_image_file);
        cols_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    int count = 0;
    vector<double> *pixel_vec;
    pixel_vec = new vector<double>();
    while(fread(&ch, sizeof(ch), 1, training_image_file))
    {
        pixel = (int)(unsigned char)ch;
        //printf("%0x\n", pixel);
        pixel_vec->push_back((double)pixel/256.0);
        count ++;
        if (count == rows_num * cols_num)
        {
            training_inputs.push_back(*pixel_vec);
            pixel_vec = new vector<double>();
            count = 0;
        }
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "rows num = " << rows_num << endl;
    cout << "cols num = " << cols_num << endl;
    cout << "inputs num = " << training_inputs.size() << endl << endl;

    fclose(training_image_file);

    magic_num = 0;
    image_num = 0;
    FILE *training_label_file = fopen("./data/train-labels.idx1-ubyte", "rb");
    cout << "loading the train labels ... ..." << endl;
    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_label_file);
        magic_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, training_label_file);
        image_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    count = 0;
    while(fread(&ch, sizeof(ch), 1, training_label_file))
    {
        label = (int)(unsigned char)ch;
        //printf("%0x\n", label);
        training_outputs.push_back((double)label);
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "labels num = " << training_outputs.size() << endl;

    fclose(training_label_file);
}

/*************************************************
Function: load_testing_data()
Description: load testing data from
Input:
    vector<vector<int> > testing_inputs
    vector<int> testing_outputs
Output: the info of testing data
Return:
Others: update the testing_inputs and testing_outputs
*************************************************/
void load_testing_data(vector<vector<double> > &testing_inputs, vector<double> &testing_outputs)
{
    char ch;
    int data_size = 4, magic_num = 0, image_num = 0, rows_num = 0, cols_num = 0, pixel, label;
    FILE *testing_image_file = fopen("./data/t10k-images.idx3-ubyte", "rb");
    cout << endl << "loading the test images ... ..." << endl;
    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_image_file);
        magic_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_image_file);
        image_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_image_file);
        rows_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_image_file);
        cols_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    int count = 0;
    vector<double> *pixel_vec;
    pixel_vec = new vector<double>();
    while(fread(&ch, sizeof(ch), 1, testing_image_file))
    {
        pixel = (int)(unsigned char)ch;
        //printf("%0x\n", pixel);
        pixel_vec->push_back((double)pixel/256.0);
        count ++;
        if (count == rows_num * cols_num)
        {
            testing_inputs.push_back(*pixel_vec);
            pixel_vec = new vector<double>();
            count = 0;
        }
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "rows num = " << rows_num << endl;
    cout << "cols num = " << cols_num << endl;
    cout << "inputs num = " << testing_inputs.size() << endl << endl;

    fclose(testing_image_file);

    magic_num = 0;
    image_num = 0;
    FILE *testing_label_file = fopen("./data/t10k-labels.idx1-ubyte", "rb");
    cout << "loading the test labels ... ..." << endl;
    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_label_file);
        magic_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    for (int i = 1; i <= data_size; i++)
    {
        fread(&ch, sizeof(ch), 1, testing_label_file);
        image_num += (int)(unsigned char)ch * pow(256, data_size-i);
    }

    count = 0;
    while(fread(&ch, sizeof(ch), 1, testing_label_file))
    {
        label = (int)(unsigned char)ch;
        //printf("%0x\n", label);
        testing_outputs.push_back((double)label);
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "labels num = " << testing_outputs.size() << endl;

    fclose(testing_label_file);
}

/*************************************************
Function: have_arg()
Description: acquire the value for the given arg name
Input:
    char *str: the given arg name
    int argc: the number of the arguments
    char **argv: the arguments
Output:
Return:
    int index: the index of the value for the given arg name
Others:
*************************************************/
int have_arg(char *str, int argc, char**argv)
{
    for (int index = 0; index < argc; index++)
    {
        if (!strcmp(str, argv[index]))
        {
            if (index == argc - 1)
            {
                cout << "no such argument!" << endl;
                exit(1);
            }
            return index;
        }
    }
    return -1;
}

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
    vector<double> training_outputs;
    vector<vector<double> > testing_inputs;
    vector<double> testing_outputs;
    int nepochs = 30;
    int batch_size = 10;
    double rate = 3.0;

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

    cout << "layers num = " << layers_num << endl;
    cout << "size = ";
    for (unsigned int i = 0; i < nn_size.size(); i++)
        cout << nn_size[i] << " ";
    cout << endl;
    cout << "nepochs = " << nepochs << endl;
    cout << "batch_size = " << batch_size << endl;
    cout << "learning rate = " << rate << endl;

    load_training_data(training_inputs, training_outputs);
    load_testing_data(testing_inputs, testing_outputs);

    Neural_Network *neural_network = new Neural_Network(nn_size);
    neural_network->SGD(training_inputs, training_outputs, nepochs,
                            batch_size, rate, testing_inputs, testing_outputs);

    return 0;
}
