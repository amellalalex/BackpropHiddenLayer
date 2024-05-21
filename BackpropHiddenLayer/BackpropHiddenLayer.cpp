// BackpropHiddenLayer.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <Eigen/Dense>

// Returns the sigmoid function value \phi(x)
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Returns the instantaneous error energy xi_j(n)
// for the neuron j having error value e_j(n)
// \xi_j(n) = (1/2) * e_j(n)^2.
float xi(float e) {
    return 0.5f * std::powf(e, 2.0f);
}

// Perceptron<# Inputs>
template<int I> class Perceptron {
public:
    Perceptron(void);

    float Evaluate(Eigen::Vector<float, I>& inputs);
    float ErrorOf(Eigen::Vector<float, I>& inputs, float expected);

private:
    Eigen::Vector<float, I> weights;
};

// Randomly initializes the weights of the perceptron
template<int I> Perceptron<I>::Perceptron(void) {
    srand(time(NULL) + rand());
    this->weights.setRandom();
}

// 'Excites' neuron using fitting input vector.
// Passes output through activation function (sigmoid).
// Returns neuron's scalar output value.
template<int I> float Perceptron<I>::Evaluate(Eigen::Vector<float, I>& inputs) {
    float sum_of_products = inputs.dot(weights);
    return sigmoid(sum_of_products);
}

// Computes the error of the evaluated input WRT the
// expected value. e_k(n) = d_k(n) - y_k(n).
template<int I> float Perceptron<I>::ErrorOf(
    Eigen::Vector<float, I>& inputs, 
    float expected
) {
    return (expected - this->Evaluate(inputs));
}

template<int N, int I> class NeuralLayer {
public:
    Eigen::Vector<float, N> Evaluate(Eigen::Vector<float, I>& inputs);
    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, I>& inputs, 
        Eigen::Vector<float, N>& expecteds
    );

private:
    std::array<Perceptron<I>, N> neurons;
};

// NeuralLayer<# of Neurons, # of Inputs/Neuron>
template<int N, int I>
Eigen::Vector<float, N> NeuralLayer<N, I>::Evaluate(
    Eigen::Vector<float, I>& inputs
)
{
    Eigen::Vector<float, N> outputs;
    for (int x = 0; x < this->neurons.size(); x++) {
        outputs(x) = this->neurons[x].Evaluate(inputs);
    }
    return outputs;
}

template<int N, int I>
Eigen::Vector<float, N> NeuralLayer<N, I>::ErrorOf(
    Eigen::Vector<float, I>& inputs,
    Eigen::Vector<float, N>& expecteds
) {
    return (expecteds - this->Evaluate(inputs));
}

// template <# of layers, # of Neurons/Layer>
// NOTE: # of Neurons/Layer == # of Inputs/Neuron
template<int L, int N> class NeuralNet {
public:
    Eigen::Vector<float, N> Evaluate(Eigen::Vector<float, N>& inputs);
    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds
    );

private:
    std::array<NeuralLayer<N, N>, L> layers;
};

template<int L, int N> 
Eigen::Vector<float, N> NeuralNet<L, N>::Evaluate(
    Eigen::Vector<float, N>& inputs
) {
    Eigen::Vector<float, N> next_input = inputs;
    for (int x = 0; x < this->layers.size(); x++) {
        next_input = this->layers[x].Evaluate(next_input);
    }
    /* next_input is now the output of the last layer in the net */

    return next_input;
}

template<int L, int N>
Eigen::Vector<float, N> NeuralNet<L, N>::ErrorOf(
    Eigen::Vector<float, N>& inputs,
    Eigen::Vector<float, N>& expecteds
) {
    return (expecteds - this->Evaluate(inputs));
}

int main(void) {
    /* const int num_inputs = 3; */
    const int num_neurons = 3;
    const int num_layers = 40;

    /* Setup */
    Eigen::Vector<float, num_neurons> input = { 1, 0, 0 };
    Eigen::Vector<float, num_neurons> expected = { 0, 0, 1 };

	/* Eval */
	NeuralNet<num_layers, num_neurons> NET;
	std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;
	std::cout << "NET.ErrorOf(input, expected) = " << std::endl << NET.ErrorOf(input, expected) << std::endl;

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
*/