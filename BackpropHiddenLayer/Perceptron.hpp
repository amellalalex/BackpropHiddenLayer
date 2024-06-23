#ifndef CSCI4156_ASSIGN1_PERCEPTRON_HPP
#define CSCI4156_ASSIGN1_PERCEPTRON_HPP

#include <Eigen/Dense>

#include "Util.hpp"

// Perceptron<# Inputs>
template<int I> class Perceptron {
    static_assert(I > 0, "Invalid Perceptron # of Inputs");
public:
    // Randomly initializes the weights of the perceptron
    Perceptron(void) {
        srand(time(NULL) + rand());
        this->weights.setRandom();
    }

    // 'Excites' neuron using fitting input vector.
    // Passes output through activation function (sigmoid).
    // Returns neuron's scalar output value.
    float Evaluate(Eigen::Vector<float, I>& inputs) const {
        return sigmoid(this->sum_of_products(inputs));
    }

    // Computes the error of the evaluated input WRT the
    // expected value. e_k(n) = d_k(n) - y_k(n).
    float ErrorOf(
        Eigen::Vector<float, I>& inputs, 
        float expected
    ) const {
        return (expected - this->Evaluate(inputs));
    }

    // Computes the instantaneous error energy of the
    // evaluated input WRT the expected value.
    // \xi_j(n) = (1/2) * e_j(n)^2.
    float ErrorEnergyOf(
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
    void LearnWithExpected(
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
    void LearnWithBackprop(
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

        /* Apply weight corrections */
        this->weights += weight_corrections;
    }

    // Computes the local gradient at this neuron ASSUMING
    // it belongs to the output layer of the Network.
    float GetLocalGradientAsOutput(
            Eigen::Vector<float, I>& inputs, 
            float expected
    ) const {
        return this->local_gradient(inputs, expected);
    }

    float GetLocalGradientForBackprop(
            Eigen::Vector<float, I>& inputs,
            float k_local_gradient,
            Eigen::Vector<float, I>& k_weights
    ) const {
        return this->local_gradient(inputs, k_local_gradient, k_weights);
    }

    Eigen::Vector<float, I> GetWeights(void) const {
        return this->weights;
    }

private:
    Eigen::Vector<float, I> weights;

    float sum_of_products(Eigen::Vector<float, I>& inputs) const {
        return inputs.dot(weights);
    }

    float local_gradient(
        Eigen::Vector<float, I>& inputs, 
        float expected
    ) const {
        return 
            this->ErrorOf(inputs, expected) 
                * sigmoid_prime(this->sum_of_products(inputs))
            ;
    }

    float local_gradient(
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

};

#endif // CSCI4156_ASSIGN1_PERCEPTRON_HPP
