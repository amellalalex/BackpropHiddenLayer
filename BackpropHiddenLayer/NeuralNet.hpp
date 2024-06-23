#ifndef CSCI4156_ASSIGN1_NEURALNET_H
#define CSCI4156_ASSIGN1_NEURALNET_H

#include <Eigen/Dense>
#include "NeuralLayer.hpp"

// template <# of layers, # of Neurons/Layer>
// NOTE: # of Neurons/Layer == # of Inputs/Neuron
template<int L, int N> class NeuralNet {
    static_assert(
        L > 0 && N > 0, 
        "Invalid NeuralNet # of Layers OR # of Neurons"
    );
public:
    Eigen::Vector<float, N> Evaluate(
        Eigen::Vector<float, N>& inputs
    ) const {
        Eigen::Vector<float, N> next_input = inputs;
        for (int x = 0; x < this->layers.size(); x++) {
            next_input = this->layers[x].Evaluate(next_input);
        }
        /* next_input is now the output of the last layer in the net */

        return next_input;
    }

    Eigen::Vector<float, N> ErrorOf(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const {
        return (expecteds - this->Evaluate(inputs));
    }

    // Returns the total instantaneous error energy
    // of the _output layer_ of the network.
    float TotalErrorEnergyOf(
        Eigen::Vector<float, N>& inputs,
        Eigen::Vector<float, N>& expecteds
    ) const {
        return this->layers.back().TotalErrorEnergyOf(inputs, expecteds);
        /* There will always be at least 1 layer due to static assertion in class def */
    }

    void BackpropWith(
            Eigen::Vector<float, N>& inputs,
            Eigen::Vector<float, N>& expecteds,
            float learning_rate
    ) {
        /* Save output layer characteristics */

        Eigen::Vector<float, N> k_local_gradients = 
            this->layers.back().GetLocalGradientsAsOutput(
                inputs,
                expecteds
            );

        Eigen::Matrix<float, N, N> k_weightss = 
            this->layers.back().GetWeightss();

        /* Apply Corrections to Output Layer */
        this->layers.back().LearnWithExpected(
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
                this->layers[x].GetLocalGradientsForBackprop(
                    inputs,
                    k_local_gradients,
                    k_weightss
                );
            next_weightss = this->layers[x].GetWeightss();

            /* Apply corrections using backprop */
            this->layers[x].LearnWithBackprop(
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

private:
    std::array<NeuralLayer<N, N>, L> layers;
};

#endif // CSCI4156_ASSIGN1_NEURALNET_H
