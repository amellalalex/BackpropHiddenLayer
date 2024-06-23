#ifndef CSCI4156_ASSIGN1_UTIL_HPP
#define CSCI4156_ASSIGN1_UTIL_HPP

#include <cmath>

// Returns the sigmoid function value \phi(x)
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float sigmoid_prime(float x) {
    return (std::exp(-x) / std::pow( 1.0f + std::exp(-x) , 2.0f));
}

// Returns the instantaneous error energy xi_j(n)
// for the neuron j having error value e_j(n)
// \xi_j(n) = (1/2) * e_j(n)^2.
inline float xi(float e) {
    return 0.5f * std::powf(e, 2.0f);
}

#endif // CSCI4156_ASSIGN1_UTIL_HPP
