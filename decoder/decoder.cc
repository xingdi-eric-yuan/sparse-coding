// Sparse Coding
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
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

#define elif else if
#define block_size 8
#define WIDTH 384
#define HEIGHT 384
#define DICT_COST 0
#define H_COST 1
#define DICT_SIZE 49
#define DATA_SIZE 142129
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
    //x = shuffle(x, 1);
}

void
readDict(mat &dict, string path){
    //read MNIST iamge into Arma Mat 
    dict = zeros<mat>(block_size * block_size, DICT_SIZE);
    FILE *streamX;
    streamX = fopen(path.c_str(), "r");
    double tpdouble;
    int counter = 0;
    while(1){
        if(fscanf(streamX, "%lf", &tpdouble) == EOF) break;
        dict(counter / DICT_SIZE, counter % DICT_SIZE) = tpdouble;
        ++ counter;
    }
    fclose(streamX);
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

double
trainingH(mat &x, mat &dict, mat &h, double lambda, double epsilon, double gamma){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;
    int dictsize = dict.n_cols;

    // define the velocity vectors.
    mat hGrad = zeros(h.n_rows, h.n_cols);
    mat dictGrad = zeros(dict.n_rows, dict.n_cols);
    mat inc_h = zeros(h.n_rows, h.n_cols);

    double lrate = 0.1; //Learning rate for weights 
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
        if(counter >= 600) break;
        ++ counter;
    }
    return cost;
}

mat
decoder(mat &x, mat &dict){
    int nsamples = x.n_cols;
    int nfeatures = x.n_rows;

    double lambda = 5e-5;  // L1-regularisation parameter (on features)
    double epsilon = 1e-5; // L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
    double gamma = 1e-2;   // L2-regularisation parameter (on basis)
    mat res = zeros(HEIGHT, WIDTH);
    mat cover = zeros(HEIGHT, WIDTH);
    for(int i = 0; i < nsamples; i++){
        
        mat xp = x.col(i);
        mat h = dict.t() * xp;
        for(int j = 0; j < h.n_rows; j++){
            h.row(j) = h.row(j) / norm(dict.col(j), 2);
        }
        double cost = trainingH(xp, dict, h, lambda, epsilon, gamma);
        cout<<"Column: "<<i<<", cost value = "<<cost<<endl;
        mat tmp = dict * h;
        tmp.reshape(block_size, block_size);
        int p_x = i % (WIDTH - block_size + 1);
        int p_y = i / (WIDTH - block_size + 1);
        res.submat(p_y, p_x, p_y + block_size - 1, p_x + block_size - 1) += tmp;
        cover.submat(p_y, p_x, p_y + block_size - 1, p_x + block_size - 1) += 1.0;
    }
    res /= cover;
    return res;
}

int 
main(int argc, char** argv){

    long start, end;
    start = clock();

    mat trainX;
    readData(trainX, "data.txt");
    cout<<"Read trainX successfully, including "<<trainX.n_rows<<" features and "<<trainX.n_cols<<" samples."<<endl;
    // Finished reading data
    mat dict;
    readDict(dict, "dict_199.txt");
    mat res = decoder(trainX, dict);
    save2txt(res, 8888);

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}
