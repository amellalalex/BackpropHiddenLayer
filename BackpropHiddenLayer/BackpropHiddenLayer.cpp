﻿// BackpropHiddenLayer.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <Eigen/Dense>

// Returns the sigmoid function value \phi(x)
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_prime(float x) {
    return (std::exp(-x) / std::pow( 1.0f + std::exp(-x) , 2.0f));
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

    void LearnWithExpected(
        Eigen::Vector<float, I>& inputs, 
        float expected,
        float learning_rate 
    );
    void LearnWithBackprop(
        Eigen::Vector<float, I>& inputs,
        float                    learning_rate,
        float                    k_local_gradient,
        Eigen::Vector<float, I>& k_weights
    );

    float GetLocalGradientAsOutput(
            Eigen::Vector<float, I>& inputs, 
            float expected
    ) const;
    float GetLocalGradientForBackprop(
            Eigen::Vector<float, I>& inputs,
            float k_local_gradient,
            Eigen::Vector<float, I>& k_weights
    ) const;

    Eigen::Vector<float, I> GetWeights(void) const;

private:
    Eigen::Vector<float, I> weights;

    float sum_of_products(Eigen::Vector<float, I>& inputs) const; // AKA v(n)

    float local_gradient(Eigen::Vector<float, I>& inputs, float expected) const;
    float local_gradient(
            Eigen::Vector<float, I>& inputs, 
            float k_local_gradient,
            Eigen::Vector<float, I>& k_weights
    ) const;
};

template<int I> float Perceptron<I>::sum_of_products(Eigen::Vector<float, I>& inputs) const {
    return inputs.dot(weights);
}

template<int I> float Perceptron<I>::local_gradient(
    Eigen::Vector<float, I>& inputs, 
    float expected
) const {
    return 
        this->ErrorOf(inputs, expected) 
            * sigmoid_prime(this->sum_of_products(inputs))
        ;
}

template<int I> float Perceptron<I>::local_gradient(
    Eigen::Vector<float, I>& inputs, 
    float k_local_gradient,
    Eigen::Vector<float, I>& k_weights
) const {

    /* Compute SUM OF ( \delta_k(n) * w_k_j(n) ) */
    float sum_of_k_local_gradient_and_weights = 0.0f;
    for(int x = 0; x < k_weights.cols(); x++) {
        sum_of_k_local_gradient_and_weights += k_local_gradient * k_weights(x);
    }

    /* Return local gradient */
    return
        sigmoid_prime(this->sum_of_products(inputs))
            * sum_of_k_local_gradient_and_weights
        ;
}

// Randomly initializes the weights of the perceptron
template<int I> Perceptron<I>::Perceptron(void) {
    srand(time(NULL) + rand());
    this->weights.setRandom();
}

// 'Excites' neuron using fitting input vector.
// Passes output through activation function (sigmoid).
// Returns neuron's scalar output value.
template<int I> float Perceptron<I>::Evaluate(Eigen::Vector<float, I>& inputs) const {
    return sigmoid(this->sum_of_products(inputs));
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

// Applies weight changes to this neuron by 
// locally calculating the error based on 
// the expected value and the evaluated 
// inputs, factoring in the provided learning
// rate in the process.
// \Delta w_j_i(n) = \eta * \delta_j(n) * y_i(n)
template<int I>
void Perceptron<I>::LearnWithExpected(
    Eigen::Vector<float, I>& inputs, 
    float expected,
    float learning_rate 
) {
    /* Start with local gradient */
    float local_gradient = this->local_gradient(inputs, expected);

    /* Calculate weight corrections */
    Eigen::Vector<float, I> weight_corrections;
    for(int x = 0; x < inputs.size(); x++) {
        weight_corrections(x) = learning_rate * local_gradient * inputs(x);
    }

    /* Apply weight corrections */
    this->weights += weight_corrections;
}

// Applies weight changes to this neuron by 
// deriving the local gradient from the k neuron
// 'ahead' of it, using its k_local_gradient and
// k_weights as it would be done if this neuron
// were part of a hidden layer. Using the local
// gradient, weight updates are applied provided
// the inputs to this neuron.
template<int I>
void Perceptron<I>::LearnWithBackprop(
    Eigen::Vector<float, I>& inputs,
    float                    learning_rate,
    float                    k_local_gradient,
    Eigen::Vector<float, I>& k_weights
) {

    /* Compute local gradient */
    float local_gradient = 
        this->local_gradient(inputs, k_local_gradient, k_weights)
        ;

    /* Calculate weight corrections */
    Eigen::Vector<float, I> weight_corrections;
    for(int x = 0; x < inputs.size(); x++) {
        weight_corrections(x) = learning_rate * local_gradient * inputs(x);
    }

    /* DEBUG */
    std::cout << "We are about to apply the following weight corrections:" << std::endl;
    std::cout << weight_corrections << std::endl << std::endl;
    std::cout << "Weights BEFORE the correction:" << std::endl;
    std::cout << this->weights << std::endl << std::endl;

    /* Apply weight corrections */
    this->weights += weight_corrections;

    /* DEBUG */
    std::cout << "Weights AFTER the correction:" << std::endl;
    std::cout << this->weights << std::endl << std::endl;
}

// Computes the local gradient at this neuron ASSUMING
// it belongs to the output layer of the Network.
template<int I>
float Perceptron<I>::GetLocalGradientAsOutput(
        Eigen::Vector<float, I>& inputs, 
        float expected
) const {
    return this->local_gradient(inputs, expected);
}

template<int I>
float Perceptron<I>::GetLocalGradientForBackprop(
        Eigen::Vector<float, I>& inputs,
        float k_local_gradient,
        Eigen::Vector<float, I>& k_weights
) const {
    return this->local_gradient(inputs, k_local_gradient, k_weights);
}

template<int I>
Eigen::Vector<float, I> Perceptron<I>::GetWeights(void) const {
    return this->weights;
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

    void LearnWithExpected(
        Eigen::Vector<float, I>& inputs, 
        Eigen::Vector<float, N>& expecteds,
        float learning_rate 
    );
    void LearnWithBackprop(
        Eigen::Vector<float, I>&    inputs,
        float                       learning_rate,
        Eigen::Vector<float, N>&    k_local_gradients,
        Eigen::Matrix<float, N, I>& k_weightss
    );

    Eigen::Vector<float, N> GetLocalGradientsAsOutput(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const;

    Eigen::Vector<float, N> GetLocalGradientsForBackprop(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& k_local_gradients,
        Eigen::Matrix<float, N, I> k_weightss
    ) const;

    Eigen::Matrix<float, N, I> GetWeightss(void) const;

private:
    std::array<Perceptron<I>, N> neurons;
};

// NeuralLayer<# of Neurons, # of Inputs/Neuron>
template<int N, int I>
Eigen::Vector<float, N> NeuralLayer<N, I>::Evaluate(
    Eigen::Vector<float, I>& inputs
) const {
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
) const {
    return (expecteds - this->Evaluate(inputs));
}

template<int N, int I>
float NeuralLayer<N, I>::TotalErrorEnergyOf(
    Eigen::Vector<float, I>& inputs,
    Eigen::Vector<float, N>& expecteds
) const {
    float sum = 0;
    for (int x = 0; x < this->neurons.size(); x++) {
        sum += ((Perceptron<I>) this->neurons[x]).ErrorEnergyOf(inputs, expecteds(x));
    }
    return sum;
}

template<int N, int I>
void NeuralLayer<N,I>::LearnWithExpected(
    Eigen::Vector<float, I>& inputs, 
    Eigen::Vector<float, N>& expecteds,
    float learning_rate 
) {
    for(int x = 0; x < this->neurons.size(); x++) {
        ((Perceptron<I>)this->neurons[x]).LearnWithExpected(
            inputs,
            expecteds(x),
            learning_rate
        );
    }
}

template<int N, int I>
void NeuralLayer<N,I>::LearnWithBackprop(
    Eigen::Vector<float, I>&    inputs,
    float                       learning_rate,
    Eigen::Vector<float, N>&    k_local_gradients,
    Eigen::Matrix<float, N, I>& k_weightss
) {
    for(int x = 0; x < this->neurons.size(); x++) {
        /* Get k_weights for this neuron */
        Eigen::Vector<float, I> k_weights = k_weightss.row(x);
        /* Apply backprop learning to this layer */
        ((Perceptron<I>)this->neurons[x]).LearnWithBackprop(
            inputs,
            learning_rate,
            k_local_gradients(x),
            k_weights
        );
    }
}

template<int N, int I>
Eigen::Vector<float, N> NeuralLayer<N, I>::GetLocalGradientsAsOutput(
    Eigen::Vector<float, I>& inputs,
    Eigen::Vector<float, N>& expecteds
) const {
    Eigen::Vector<float, N> local_gradients;
    for(int x = 0; x < this->neurons.size(); x++) {
        local_gradients(x) = 
            ((Perceptron<I>)this->neurons[x]).GetLocalGradientAsOutput(
                inputs, 
                expecteds(x)
            );
    }
    return local_gradients;
}

template<int N, int I>
Eigen::Vector<float, N> NeuralLayer<N, I>::GetLocalGradientsForBackprop(
    Eigen::Vector<float, I>& inputs,
    Eigen::Vector<float, N>& k_local_gradients,
    Eigen::Matrix<float, N, I> k_weightss
) const {
    Eigen::Vector<float, N> local_gradients;
    for(int x = 0; x < this->neurons.size(); x++) {
        /* Get k_weights for this neurons */
        Eigen::Vector<float, I> k_weights = k_weightss.row(x);
        /* Get local gradients for this layer */
        local_gradients(x) = 
            ((Perceptron<I>)this->neurons[x]).GetLocalGradientForBackprop(
                inputs,
                k_local_gradients(x),
                k_weights
            );
    }
    return local_gradients;
}

template<int N, int I> 
Eigen::Matrix<float, N, I> NeuralLayer<N, I>::GetWeightss(void) const {
    Eigen::Matrix<float, N, I> weightss;
    for(int x = 0; x < this->neurons.size(); x++) {
        weightss.row(x) = ((Perceptron<I>)this->neurons[x]).GetWeights();
    }
    return weightss;
}

// template <# of layers, # of Neurons/Layer>
// NOTE: # of Neurons/Layer == # of Inputs/Neuron
template<int L, int N> class NeuralNet {
    static_assert(
        L > 0 && N > 0, 
        "Invalid NeuralNet # of Layers OR # of Neurons"
    );
public:
    Eigen::Vector<float, N> Evaluate(Eigen::Vector<float, N>& inputs) const;

    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const;

    float TotalErrorEnergyOf(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const;

    void BackpropWith(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds,
        float learning_rate
    );

private:
    std::array<NeuralLayer<N, N>, L> layers;
};

template<int L, int N> 
Eigen::Vector<float, N> NeuralNet<L, N>::Evaluate(
    Eigen::Vector<float, N>& inputs
) const {
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
) const {
    return (expecteds - this->Evaluate(inputs));
}

// Returns the total instantaneous error energy
// of the _output layer_ of the network.
template<int L, int N>
float NeuralNet<L, N>::TotalErrorEnergyOf(
    Eigen::Vector<float, N>& inputs,
    Eigen::Vector<float, N>& expecteds
) const {
    return ((NeuralLayer<N, N>)this->layers.back()).TotalErrorEnergyOf(inputs, expecteds);
    /* There will always be at least 1 layer due to static assertion in class def */
}

template<int L, int N> void NeuralNet<L, N>::BackpropWith(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds,
        float learning_rate
) {
    /* Save output layer characteristics */

    Eigen::Vector<float, N> k_local_gradients = 
        ((NeuralLayer<N,N>)this->layers.back()).GetLocalGradientsAsOutput(
            inputs,
            expecteds
        );

    Eigen::Matrix<float, N, N> k_weightss = 
        ((NeuralLayer<N,N>)this->layers.back()).GetWeightss();

    /* Apply Corrections to Output Layer */
    ((NeuralLayer<N,N>)this->layers.back()).LearnWithExpected(
        inputs,
        expecteds,
        learning_rate
    );

    /* For remaining layers, apply backprop */
    Eigen::Vector<float, N> next_local_gradients = k_local_gradients;
    Eigen::Matrix<float, N, N> next_weightss = k_weightss;
    /* Treat k_*** as values to apply for x'th iteration */
    /* Treat next_*** as persistence state for (x+1)'th */

    /* Note that we are excluding the output layer */
    for(int x = this->layers.size()-2; x >= 0; x--) {
        /* Save current layer's characteristics */
        next_local_gradients = 
            ((NeuralLayer<N,N>)this->layers[x]).GetLocalGradientsForBackprop(
                inputs,
                k_local_gradients,
                k_weightss
            );
        next_weightss = ((NeuralLayer<N,N>)this->layers[x]).GetWeightss();

        /* Apply corrections using backprop */
        ((NeuralLayer<N,N>)this->layers[x]).LearnWithBackprop(
            inputs,
            learning_rate,
            k_local_gradients,
            k_weightss
        );

        /* Update x'th iteration characteristics */
        k_local_gradients = next_local_gradients;
        k_weightss = next_weightss;
    }
}

int main(void) {
    /* const int num_inputs = 3; */
    const int num_neurons = 3;
    const int num_layers = 3;
    const float learning_rate = 0.3f;

    /* Setup */
    Eigen::Vector<float, num_neurons> input = { 1, 0, 0 };
    Eigen::Vector<float, num_neurons> expected = { 0, 0, 1 };

    /* Eval */
    NeuralNet<num_layers, num_neurons> NET;
    std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;
    std::cout << "NET.ErrorOf(input, expected) = " << std::endl << NET.ErrorOf(input, expected) << std::endl;
    std::cout << "NET.TotalErrorEnergyOf(input, expected) = " << std::endl << NET.TotalErrorEnergyOf(input, expected) << std::endl;

    /* Backprop */
    for(;;) {
        NET.BackpropWith(input, expected, learning_rate);
        std::cout << "NET.BackpropWith(input, expected, learning_rate);" << std::endl;
        std::cout << "NET.Evaluate(input) = " << std::endl << NET.Evaluate(input) << std::endl;
        (void)getc(stdin);
    }

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
