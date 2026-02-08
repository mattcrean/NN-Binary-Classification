#include <iostream>
#include <vector>
#include "project2_a.h"

// Driver code for Analysis Task 2
//
// Uses spiral_training_data to build a neural network
// of (2,4,4,1) to be varied for analysis 
// with tanh activation function,
// learning rate 0.1, target cost 1x10^-3 and
// max_iter 4x10^6

using namespace BasicDenseLinearAlgebra;

int main()
{
    ActivationFunction* tanh = new TanhActivationFunction;

    // Define layers
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
    unsigned n_neuron = 4;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));
    n_neuron = 4;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));
    n_neuron = 1;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));

    // Build network
    NeuralNetwork network(n_input, non_input_layer);

    // Read training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    network.read_training_data("spiral_training_data.dat", training_data);

    // Initialise parameters with given normal distributuon
    network.initialise_parameters(0.0, 0.1);
    const double eta = 0.01;
    const double tol = 1e-3;
    const unsigned max_iter = 4e6;

    // Train the neural network
    network.train(training_data, eta, tol, max_iter, "task2_conv_history.dat");

    // Define evaluation points for contour plot to [0,1]x[0,1]
    // with step size 0.01
    std::vector<DoubleVector> grid_points;
    double step = 0.01;
    for(double x = 0.0; x <= 1.0 + 1e-12; x += step)
    {
        for(double y = 0.0; y <= 1.0 + 1e-12; y += step)
        {
            DoubleVector p(2);
            p[0] = x;
            p[1] = y;
            grid_points.push_back(p);
        }
    }

    // Save parameters for analysis
    network.output("task2_network_data.dat", grid_points);

    return 0;
}
