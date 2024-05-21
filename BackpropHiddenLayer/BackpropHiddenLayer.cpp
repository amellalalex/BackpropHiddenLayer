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
        static_assert(I > 0, "Invalid Perceptron # of Inputs");
public:
    Perceptron(void);

    float Evaluate(Eigen::Vector<float, I>& inputs) const;
    float ErrorOf(Eigen::Vector<float, I>& inputs, float expected) const;
    float ErrorEnergyOf(Eigen::Vector<float, I>& inputs, float expected) const;

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
template<int I> float Perceptron<I>::Evaluate(Eigen::Vector<float, I>& inputs) const {
    float sum_of_products = inputs.dot(weights);
    return sigmoid(sum_of_products);
}

// Computes the error of the evaluated input WRT the
// expected value. e_k(n) = d_k(n) - y_k(n).
template<int I> float Perceptron<I>::ErrorOf(
    Eigen::Vector<float, I>& inputs, 
    float expected
) const {
    return (expected - this->Evaluate(inputs));
}

// Computes the instantaneous error energy of the
// evaluated input WRT the expected value.
// \xi_j(n) = (1/2) * e_j(n)^2.
template<int I> 
float Perceptron<I>::ErrorEnergyOf(
    Eigen::Vector<float, I>& inputs, float expected
) const {
    return xi(this->ErrorOf(inputs, expected));
}

template<int N, int I> class NeuralLayer {
        static_assert(
                N > 0 && I > 0, 
                "Invalid NeuralLayer # of Neurons OR # of Inputs"
        );
public:
    Eigen::Vector<float, N> Evaluate(Eigen::Vector<float, I>& inputs) const;

    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, I>& inputs, 
        Eigen::Vector<float, N>& expecteds
    ) const;

    float TotalErrorEnergyOf(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const;

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

template<int N, int I>
float NeuralLayer<N, I>::TotalErrorEnergyOf(
    Eigen::Vector<float, I>& inputs,
    Eigen::Vector<float, N>& expecteds
) {
    float sum = 0;
    for (int x = 0; x < this->neurons.size(); x++) {
        sum += ((Perceptron<I>) this->neurons[x]).ErrorEnergyOf(inputs, expecteds(x));
    }
    return sum;
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
    float TotalErrorEnergyOf(
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

// Returns the total instantaneous error energy
// of the _output layer_ of the network.
template<int L, int N>
float NeuralNet<L, N>::TotalErrorEnergyOf(
    Eigen::Vector<float, N>& inputs,
    Eigen::Vector<float, N>& expecteds
) {
    if (this->layers.size() > 0) {
        return ((NeuralLayer<N, N>)this->layers.back()).TotalErrorEnergyOf(inputs, expecteds);
    }
    else {
        return 0.0f;
    }
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
	std::cout << "NET.TotalErrorEnergyOf(input, expected) = " << std::endl << NET.TotalErrorEnergyOf(input, expected) << std::endl;

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
*/
