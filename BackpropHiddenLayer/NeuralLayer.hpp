#ifndef CSCI4156_ASSIGN1_NEURALLAYER_HPP
#define CSCI4156_ASSIGN1_NEURALLAYER_HPP

#include <Eigen/Dense>
#include "Perceptron.hpp"

template<int N, int I> class NeuralLayer {
    static_assert(
        N > 0 && I > 0, 
        "Invalid NeuralLayer # of Neurons OR # of Inputs"
    );
public:
    // NeuralLayer<# of Neurons, # of Inputs/Neuron>
    Eigen::Vector<float, N> Evaluate(
        Eigen::Vector<float, I>& inputs
    ) const {
        Eigen::Vector<float, N> outputs;
        for (int x = 0; x < this->neurons.size(); x++) {
            outputs(x) = this->neurons[x].Evaluate(inputs);
        }
        return outputs;
    }

    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const {
        return (expecteds - this->Evaluate(inputs));
    }

    float TotalErrorEnergyOf(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const {
        float sum = 0;
        for (int x = 0; x < this->neurons.size(); x++) {
            sum += this->neurons[x].ErrorEnergyOf(inputs, expecteds(x));
        }
        return sum;
    }

    void LearnWithExpected(
        Eigen::Vector<float, I>& inputs, 
        Eigen::Vector<float, N>& expecteds,
        float learning_rate 
    ) {
        for(int x = 0; x < this->neurons.size(); x++) {
            this->neurons[x].LearnWithExpected(
                inputs,
                expecteds(x),
                learning_rate
            );
        }
    }

    void LearnWithBackprop(
        Eigen::Vector<float, I>&    inputs,
        float                       learning_rate,
        Eigen::Vector<float, N>&    k_local_gradients,
        Eigen::Matrix<float, N, I>& k_weightss
    ) {
        for(int x = 0; x < this->neurons.size(); x++) {
            /* Get k_weights for this neuron */
            Eigen::Vector<float, I> k_weights = k_weightss.row(x);
            /* Apply backprop learning to this layer */
            this->neurons[x].LearnWithBackprop(
                inputs,
                learning_rate,
                k_local_gradients(x),
                k_weights
            );
        }
    }

    Eigen::Vector<float, N> GetLocalGradientsAsOutput(
        Eigen::Vector<float, I>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const {
        Eigen::Vector<float, N> local_gradients;
        for(int x = 0; x < this->neurons.size(); x++) {
            local_gradients(x) = 
                this->neurons[x].GetLocalGradientAsOutput(
                    inputs, 
                    expecteds(x)
                );
        }
        return local_gradients;
    }

    Eigen::Vector<float, N> GetLocalGradientsForBackprop(
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
                this->neurons[x].GetLocalGradientForBackprop(
                    inputs,
                    k_local_gradients(x),
                    k_weights
                );
        }
        return local_gradients;
    }

    Eigen::Matrix<float, N, I> GetWeightss(void) const {
        Eigen::Matrix<float, N, I> weightss;
        for(int x = 0; x < this->neurons.size(); x++) {
            weightss.row(x) = this->neurons[x].GetWeights();
        }
        return weightss;
    }

private:
    std::array<Perceptron<I>, N> neurons;
};

#endif // CSCI4156_ASSIGN1_NEURALLAYER_HPP
