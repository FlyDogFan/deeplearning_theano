/*************************************************
Author: Tianwen Jiang
Date: 2016-01-27
Description: load the hand-wrte data
**************************************************/

#ifndef LOAD_HANDWRITE_DATA_H_INCLUDED
#define LOAD_HANDWRITE_DATA_H_INCLUDED

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

/*************************************************
Function: load_training_data()
Description: load training data from
Input:
    vector<vector<double> > training_inputs
    vector<vector<double> > training_outputs
Output: the info of training data
Return:
Others: update the training_inputs and training_outputs
*************************************************/
void load_training_data(vector<vector<double> > &training_inputs, vector<vector<double> > &training_outputs)
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

    vector<double> label_vec;
    while(fread(&ch, sizeof(ch), 1, training_label_file))
    {
        label = (int)(unsigned char)ch;
        for (int i = 0; i < 10; i++)
        {
            if (label == i)
                label_vec.push_back(1.0);
            else
                label_vec.push_back(0.0);
        }
        training_outputs.push_back(label_vec);
        label_vec.clear();
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
    vector<vector<int> > test_inputs
    vector<int> test_outputs
Output: the info of testing data
Return:
Others: update the test_inputs and test_outputs
*************************************************/
void load_testing_data(vector<vector<double> > &test_inputs, vector<vector<double> > &test_outputs)
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
            test_inputs.push_back(*pixel_vec);
            pixel_vec = new vector<double>();
            count = 0;
        }
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "rows num = " << rows_num << endl;
    cout << "cols num = " << cols_num << endl;
    cout << "inputs num = " << test_inputs.size() << endl << endl;

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

    vector<double> label_vec;
    while(fread(&ch, sizeof(ch), 1, testing_label_file))
    {
        label = (int)(unsigned char)ch;
        for (int i = 0; i < 10; i++)
        {
            if (label == i)
                label_vec.push_back(1.0);
            else
                label_vec.push_back(0.0);
        }
        test_outputs.push_back(label_vec);
        label_vec.clear();
    }
    cout << "magic num = " << magic_num << endl;
    cout << "image num = " << image_num << endl;
    cout << "labels num = " << test_outputs.size() << endl;

    fclose(testing_label_file);
}

#endif // LOAD_HANDWRITE_DATA_H_INCLUDED
