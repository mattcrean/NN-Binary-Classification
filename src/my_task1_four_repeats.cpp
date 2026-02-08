#include <iostream>
#include <vector>
#include "project2_a.h"

// Second driver code for Analysis Task 1
// 
// Identical purpose as first driver code for
// Analysis Task 1, but
// training process is repeated four times.
// Four different files outputed for each run
// for analysis

using namespace BasicDenseLinearAlgebra;

int main()
{
    ActivationFunction* tanh = new TanhActivationFunction;

    // Define layers
    unsigned n_input = 2;
    std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
    unsigned n_neuron = 3;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));
    n_neuron = 3;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));
    n_neuron = 1;
    non_input_layer.push_back(std::make_pair(n_neuron, tanh));

    // Build network
    NeuralNetwork temp(n_input, non_input_layer);

    // Read training data
    std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
    temp.read_training_data("project_training_data.dat", training_data);

    // Parameters for network
    const double eta = 0.1;
    const double tol = 1e-4;
    const unsigned max_iter = 100000;

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

    // Repeat training process four times
    for (unsigned run = 1; run <= 4; run++)
    {
        // Define neural network
        NeuralNetwork network(n_input, non_input_layer);
        // Initialise parameters with given normal distributuon
        network.initialise_parameters(0.0, 0.1);

        // Change name of file for each run
        std::string history_file = "task1_conv_history_run" + std::to_string(run) + ".dat";
        std::string output_file = "task1_network_data_run" + std::to_string(run) + ".dat";

        // Train network and output parameters
        network.train(training_data, eta, tol, max_iter, history_file);
        network.output(output_file, grid_points);
    }

    return 0;
}
