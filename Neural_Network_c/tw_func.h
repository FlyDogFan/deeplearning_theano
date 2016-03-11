/*************************************************
Author: Tianwen Jiang
Date: 2016-01-27
Description: some usual function
**************************************************/

#ifndef TW_FUNC_H_INCLUDED
#define TW_FUNC_H_INCLUDED

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace std;

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

#endif // TW_FUNC_H_INCLUDED
