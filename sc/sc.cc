// Sparse Coding
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// To run this code, you should have Armadillo in your computer.
// Have fun with it :)
#include <armadillo>
#include <math.h>
#include <fstream>
#include <iostream>
#include <random>  

using namespace arma;
using namespace std;
#define DICT_COST 0
#define H_COST 1
#define DICT_SIZE 49
#define DATA_SIZE 142129
#define block_size 8
#define elif else if
int batch;

void
readData(mat &x, string xpath){
    //read MNIST iamge into Arma Mat 
    x = zeros<mat>(block_size * block_size, DATA_SIZE);
    FILE *streamX;
    streamX = fopen(xpath.c_str(), "r");
    double tpdouble;
    int counter = 0;
    while(1){
        if(fscanf(streamX, "%lf", &tpdouble) == EOF) break;
        x(counter / DATA_SIZE, counter % DATA_SIZE) = tpdouble;
        ++ counter;
    }
    fclose(streamX);
    x = shuffle(x, 1);
}

void
matRandomInit(mat &m, int rows, int cols, double scaler){
    m = randn<mat>(rows, cols);
    m = m * scaler;
}

void
save2txt(mat &data, int step){
    string str = "dict_";
    string s = std::to_string(step);
    str += s;
    str += ".txt";
    FILE *pOut = fopen(str.c_str(), "w");
    for(int i=0; i<data.n_rows; i++){
        for(int j=0; j<data.n_cols; j++){
            fprintf(pOut, "%lf", data(i, j));
            if(j == data.n_cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}

double
getSparseCodingCost(mat &x, mat &dict, mat &h, mat &dictGrad, mat &hGrad, double lambda, double epsilon, double gamma, int cond){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;
    int dictsize = dict.n_cols;

    mat delta = dict * h - x;
    double J1 = accu(pow(mean(delta, 1), 2.0));
    mat sparsityMatrix = sqrt(pow(h, 2.0) + epsilon);
    double J2 = lambda * accu(mean(sparsityMatrix, 1));
    double J3 = gamma * accu(pow(dict, 2.0));
    double cost = 0.0;
    if(cond == DICT_COST) cost = J1 + J3;
    elif(cond == H_COST) cost = J1 + J2;

    dictGrad = (2 * dict * h * h.t() - 2 * x * h.t()) / nsamples;
    dictGrad += 2 * gamma * dict;

    hGrad = (2 * dict.t() * dict * h - 2 * dict.t() * x) / nsamples;
    hGrad += lambda * (h / sparsityMatrix);
    return cost;
}

void
gradientChecking(mat &x, mat &dict, mat &h){

    //Gradient Checking (remember to disable this part after checking)
    double lambda = 5e-5;  // L1-regularisation parameter (on features)
    double epsilon = 1e-2; // L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
    double gamma = 1e-2;   // L2-regularisation parameter (on basis)
    mat dictGrad = zeros(dict.n_rows, dict.n_cols);
    mat hGrad = zeros(h.n_rows, h.n_cols);
    double cost = getSparseCodingCost(x, dict, h, dictGrad, hGrad, lambda, epsilon, gamma, 1);
    mat hg = hGrad;
    cout<<"Gradient Checking Now ...."<<endl;
    double e = 1e-4;
    for(int i=0; i<h.n_rows; i++){
        for(int j=0; j<h.n_cols; j++){
            double memo = h(i, j);
            h(i, j) = memo + e;
            cost = getSparseCodingCost(x, dict, h, dictGrad, hGrad, lambda, epsilon, gamma, 1);
            double value1 = cost;
            h(i, j) = memo - e;
            cost = getSparseCodingCost(x, dict, h, dictGrad, hGrad, lambda, epsilon, gamma, 1);
            double value2 = cost;
            double tp = (value1 - value2) / (2 * e);
            cout<<i<<", "<<j<<", "<<tp<<", "<<hg(i, j)<<", "<<hg(i, j) / tp<<endl;
            h(i, j) = memo;
        }
    }
}

double
trainingDict(mat &x, mat &dict, mat &h, double lambda, double epsilon, double gamma){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;
    int dictsize = dict.n_cols;

    // define the velocity vectors.
    mat dictGrad = zeros(dict.n_rows, dict.n_cols);
    mat hGrad = zeros(h.n_rows, h.n_cols);
    mat inc_dict = zeros(dict.n_rows, dict.n_cols);

    double lrate = 0.05; //Learning rate for weights 
    double weightcost = 0.0002;   
    double initialmomentum = 0.5;
    double finalmomentum = 0.9;
    double errsum = 0.0;
    double momentum;
    double cost;
    int counter = 0;

    while(1){
        if(counter > 10) momentum = finalmomentum;
        else momentum = initialmomentum;
        cost = getSparseCodingCost(x, dict, h, dictGrad, hGrad, lambda, epsilon, gamma, DICT_COST);
        // update weights 
        inc_dict = momentum * inc_dict + lrate * (dictGrad - weightcost * dict);
        dict -= inc_dict;
        if(counter >= 800) break;
        ++ counter;
    }
    cout<<"training dict, Cost function value = "<<cost<<endl;
    return cost;
}

double
trainingH(mat &x, mat &dict, mat &h, double lambda, double epsilon, double gamma){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;
    int dictsize = dict.n_cols;

    // define the velocity vectors.
    mat hGrad = zeros(h.n_rows, h.n_cols);
    mat dictGrad = zeros(dict.n_rows, dict.n_cols);
    mat inc_h = zeros(h.n_rows, h.n_cols);

    double lrate = 0.05; //Learning rate for weights 
    double weightcost = 0.0002;   
    double initialmomentum = 0.5;
    double finalmomentum = 0.9;
    double errsum = 0.0;
    double momentum;
    double cost;
    int counter = 0;

    while(1){
        if(counter > 10) momentum = finalmomentum;
        else momentum = initialmomentum;
        cost = getSparseCodingCost(x, dict, h, dictGrad, hGrad, lambda, epsilon, gamma, H_COST);
        // update weights 
        inc_h = momentum * inc_h + lrate * (hGrad - weightcost * h);
        h -= inc_h;
        if(counter >= 800) break;
        ++ counter;
    }
    cout<<"training H, Cost function value = "<<cost<<endl;
    return cost;
}

mat
dictionaryLearning(mat &x, int dictsize){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;

    double lambda = 5e-5;  // L1-regularisation parameter (on features)
    double epsilon = 1e-5; // L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
    double gamma = 1e-2;   // L2-regularisation parameter (on basis)
    mat dict, h;
    matRandomInit(dict, nfeatures, dictsize, 0.12);
    int counter = 0;
    double cost = 0;
    double last = 0;

    //Gradient Checking...
//    mat randx = x.cols(5, 10);
//    h = dict.t() * randx;
//    for(int i = 0; i < h.n_rows; i++){
//        h.row(i) = h.row(i) / norm(dict.col(i), 2);
//    }
//    gradientChecking(randx, dict, h);
    while(1){
        int randomNum = ((long)rand() + (long)rand()) % (nsamples - batch);
        mat randx = x.cols(randomNum, randomNum + batch - 1);
        h = dict.t() * randx;
        for(int i = 0; i < h.n_rows; i++){
            h.row(i) = h.row(i) / norm(dict.col(i), 2);
        }
        cout<<"counter = "<<counter<<endl;
        cost = trainingH(randx, dict, h, lambda, epsilon, gamma);
        cost = trainingDict(randx, dict, h, lambda, epsilon, gamma);
        //if(counter != 0 && fabs(cost - last) < 1e-4) break;
        if(counter >= 1000) break;
        last = cost;
        if(counter % 5 == 0)
            save2txt(dict, counter / 5);
        ++ counter;
    }
    return dict;
}

int 
main(int argc, char** argv){

    long start, end;
    start = clock();
    mat trainX;
    readData(trainX, "data.txt");

    cout<<"Read trainX successfully, including "<<trainX.n_rows<<" features and "<<trainX.n_cols<<" samples."<<endl;
    // Finished reading data
    batch = 1000;
    mat dict = dictionaryLearning(trainX, DICT_SIZE);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}
