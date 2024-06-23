// BackpropHiddenLayer.cpp : Defines the entry point for the application.
//

// Standard C Libraries
#include <cstdio>

// Standard C++ Libraries
#include <iostream>
#include <fstream>
#include <array>

// External Dependencies
#include <Eigen/Dense>

// Project Headers
#include "NeuralNet.hpp"

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> 
load_iris(const char *path) {
    // Settings
    const int num_cols = 3;

    std::ifstream file(path);
    std::string line;

    size_t  num_lines;
    float   a,b,c;
    int     class_num;

    // Get number of lines in file data 
    num_lines = 0;
    while(std::getline(file, line)) {
        ++num_lines;
    }
    
    // DEBUG
    std::cout << "num_lines = " << num_lines << std::endl;

    // Reset file pointer
    file.clear();
    file.seekg(0);

    // Create dynamic matrix per data size
    Eigen::MatrixX<float> iris_data_in(num_lines, num_cols);
    Eigen::MatrixX<int> iris_data_keys(num_lines, num_cols);
    
    // Read iris data into matrix
    for(size_t x = 0; std::getline(file, line); ++x) {
        std::sscanf(line.c_str(), "%3f,%3f,%3f,%1d", &a, &b, &c, &class_num);

        // DEBUG 
        std::cout << "a = " << a << ", b = " << b << ", c = " << c << ", class_num = " << class_num << std::endl;

        // Set instance inputs
        iris_data_in(x, 0) = a;
        iris_data_in(x, 1) = b;
        iris_data_in(x, 2) = c;

        // Set instance output class key (one hot)
        switch(class_num) {
        case 0:
            iris_data_keys(x, 0) = 1;
            iris_data_keys(x, 1) = 0;
            iris_data_keys(x, 2) = 0;
            break;
        case 1:
            iris_data_keys(x, 0) = 0;
            iris_data_keys(x, 1) = 1;
            iris_data_keys(x, 2) = 0;
            break;
        case 2:
            iris_data_keys(x, 0) = 0;
            iris_data_keys(x, 1) = 0;
            iris_data_keys(x, 2) = 1;
            break;
        }
    }

    // DEBUG
    std::cout << "Iris Values = " << std::endl << iris_data_in << std::endl;
    std::cout << "Iris Keys = " << std::endl << iris_data_keys << std::endl;

    return iris_data_in;
}

int main(void) {
    /* const int num_inputs = 3; */
    const int num_neurons = 3;
    const int num_layers = 3;
    const float learning_rate = 0.8f;
    const int series_len = 5;

    /* Setup */
    Eigen::Vector<float, num_neurons> input = { 1, 0, 0 };
    Eigen::Vector<float, num_neurons> expected = { 0, 0, 1 };

    /* Batch Setup */
    Eigen::Matrix<float, series_len, num_neurons> inputs {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 },
        { 0, 1, 0 },
        { 1, 0, 0 },
    };
    Eigen::Matrix<float, series_len, num_neurons> expecteds {
        { 0, 0, 1 },
        { 1, 0, 1 },
        { 1, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
    };

    /* Eval */
    NeuralNet<num_layers, num_neurons> NET;
    std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;
    std::cout << "NET.ErrorOf(input, expected) = " << std::endl << NET.ErrorOf(input, expected) << std::endl;
    std::cout << "NET.TotalErrorEnergyOf(input, expected) = " << std::endl << NET.TotalErrorEnergyOf(input, expected) << std::endl;

    /* Backprop */
    std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;
    std::cout << "BACKPROP START" << std::endl;
    for(int x = 0; x < series_len; x++) {
        Eigen::Vector<float, num_neurons> next_input = inputs.row(x);
        Eigen::Vector<float, num_neurons> next_expected = expecteds.row(x);

        NET.BackpropWith(next_input, next_expected, learning_rate);
        std::cout << "NET.BackpropWith(next_input, next_expected, learning_rate);" << std::endl;
        (void)getc(stdin);
    }
    std::cout << "BACKPROP END" << std::endl;
    std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;

    // DEBUG 
    load_iris("iris.data");

    return 0;
}

/**
* Some math:
* ----------
* - We have things down for feed-forward functions
* -- i.e.: Inputs enter perceptron, sum-of-products of weights x inputs passed into 
*    activation function, this value is passed as the scalar output.
* 
* - We now need to implement learning, which requires:
* -- Knowing the expected answer during Evaluate()ing Perceptrons and NeuralLayers
* --- There could be a special method for this, maybe called Learn()
* -- Comparing the expected answer against the obtained one, we can calculate
*    the error.
* -- By applying a cost function to the error (per neuron), we obtain the 
*    _Instantaneous Value of the Error Energy_ \Epsilon(n). 
* --- This is the function we are trying to minimize as we iterate and 'learn'.

* -- Lastly, 'learning' is applied via back-propagation following a stochastic
*    model of gradient descent -- in other words, the estimated instantaneous
*    variation in the cost function with respect to the paramters being adjusted.
* --- In actual English, the weight adjustment for each neuron needs to be calculated
*     and applied based on the _learning rate_, the error e_k(n) and the input x_j(n)
* --- *** NOTE: This is ONLY actually true of the visible output layer -- and true
*               exclusively when there is only one layer to the entire network.

* --- Error back-propagation learning requires every neuron in the network to do 2 things:
* ---- 1. The computation of the function signal appearing at the output of the neuron
*         being the application of the continuous non-linear function of the input signal
*         and the synaptics weights associated with that particular neuron
* ---- 2. The computed estimate of a gradient vector being the gradient of the error
*         surface with respect to the weights connected to the input of the neuron.
*         This is necessary for the backward-pass through the network (and weight updates).
*
* --- Doing a bit of partial-derivative math (as shown in Simon Haykin p.163):
* ---- d_partial(\xi(n)) / d_partial(w_j_i(n)) = - e_j(n) * \phi'_j(v_j(n)) * y_i(n)
*  OR in English:
* ---- partial der. of total error energy WRT partial der. of weight for i'th input into j'th neuron is 
*      equal to the negative error value of j'th neuron times the slope of the activation function at the
*      NET's internal signal value (sum of products of weights x inputs) times the i'th input signal 
*                                                                 (AKA output of i'th neuron of prev layer)
*
* --- Okay, so, with this gradient value, we may obtain the weight-correction for the j'th neuron with:
* ---- \Delta w_j_i(n) = \eta * d_partial(\xi(n)) / d_partial(w_j_i(n))
*   OR in English:
* ---- The change in weight at jth neuron for its i'th input = the learning rate * gradient descent value
*
* --- Paying close attention to the gradient descent formula reveals that the only change in value for
*     differing i values (that is, for a different input) is y_i(n). In other words, for a single neuron,
*     the application of its 'lesson' can be generalized across all inputs under the 'local error' or 
*     'local gradient':
* ---- \Delta w_j_i(n) = \eta * \delta_j(n) * y_i(n)
*       where \delta_j(n) is the local gradient (or local error)
*  OR in English:
* ----- The change in weight at neuron j for input i = the learning rate * local gradient * i'th input signal
* 
* --- The local gradient value may be computed on a per-neuron basis as per the following:
* ---- \delta_j(n) = e_j(n) * \phi'_j(v_j(n))
*  OR in English:
* ---- local gradient = error value at neuron j * slope of activation function AT j'th neuron internal signal
*                                                                                 AKA the sum of products neuron j
*
* NOTE: So far, we have not addressed the credit-assignment problem
* -- IOW, this only works for the output layer.
*
* --- Where j is a hidden neuron and k is a subsequent output neuron,
* ---- delta_j(n) = \phi'_j(v_j(n)) * SUM_k( \delta_k(n) * w_k_j(n) )
*  OR in English:
* ---- local gradient = slope of the activation function AT the internal signal value of neuron j
*                               * the SUM OF all k neurons ( local gradient at neuron k * the weight
*                                                           for the j'th input into the k'th neuron )
*  WHERE
* ---- \Delta w_j_i(n) = \eta * \delta_j(n) * y_i(n)
*  OR in English:
* ---- Weight correction = learning rate * local gradient * input signal i to neuron j
*/

/**
* A NOTE on the mysterious 'amnesia' I experienced in my C++ journey of ML :-)
* 
* Once upon a time...a C programmer was on a mission...to prove that they could
* tame the beast that C++ was. They wrote their entire CS assignment for a mul-
* ti-layer perceptron in C++. They were proud. Except for one little thing:
*   - Every time they ran the program, it seemed like NONE of the learning
*     was remembered by the network.
* Attempting to debug and take a closer look revealed that weights were indeed
* changing when called to learn and backpropagate. So what was going on?
* 
* Well, you see, the C programmer was used to typecasting when it was convenient.
* After all, in C, typecasting is more of a hint to the compiler than anything else.
* Sometimes it can be used cleverly to convert and manipulate datatypes in unusual 
* ways.
* 
* But in C++ -- typecasting is an entirely different beast. And when it comes to
* casting CLASSES in C++, it does something far more sinister than it did in C.
* It doesn't just provide a hint to the compiler nor help the LSP resolve symbols,
* it actually DISCONNECTS the underlying class type from the method call (thought
* to be stored in persistent memory) and INSTEAD replaced with a temporary...or
* something of that kind.
* 
* This meant that everytime the low-level, individual methods were tried for 
* debugging, it worked as intended...but when the entire system was tried in 
* its natural habitat, everything broke...
* 
* Let it be a staunch reminder and a warning: C++ TYPECASTING IS NOT FOR FUN.
*/
