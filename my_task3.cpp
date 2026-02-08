#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <chrono>
#include <ctime>
#include <random> 
#include <sstream>

// Driver code for Analysis Task 3
//
// Computational cost is analysed here by building
// a neural network, timing how long it takes for
// - a feed forward operation
// and how long it takes to compute gradients via
// - finite-differencing
// - back-propagation
// and output times to file.
//
// Here, we show keeping number of layers fixed,
// but similar code constructed to keep the
// number of neurons fixed


// Necessary for access to:
// - compute_gradients_finite_difference
// - compute_gradients_backprop
#define private public
#include "project2_a.h"
#undef private

using namespace BasicDenseLinearAlgebra;

int main()
{  
    // Open file for timing results
    std::ofstream outfile("task3_neurons.dat");

    // Define timer
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    // Number of layers
    unsigned N_layer = 10; 

    // Values of neurons to loop over
    std::vector<unsigned> neurons = {10,20,40,80,160};

    // Loop over neuron values
    for (unsigned N_neuron : neurons)
    {
        ActivationFunction* tanh = new TanhActivationFunction;

        // Construct network with desired number of neurons/layers
        std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
        for (unsigned l = 0; l < N_layer - 1; l++)
        {
            non_input_layer.push_back(std::make_pair(N_neuron, tanh));
        }
        NeuralNetwork network(N_neuron, non_input_layer);
        
        // Initialise parameters for given normal distribution
        network.initialise_parameters(0.0, 0.1);

        // Make random input x and target y
        DoubleVector x(N_neuron), y(N_neuron);
        std::normal_distribution<> normal_dist(0.0, 0.1);
        for (unsigned i = 0; i < N_neuron; i++)
        {
            x[i] = normal_dist(RandomNumber::Random_number_generator);
            y[i] = normal_dist(RandomNumber::Random_number_generator);
        }

        // Prepare for storing results
        DoubleVector out;
        std::vector<DoubleMatrix> dW;
        std::vector<DoubleVector> db;

        // Timing for feed-forward (repeated 10 times, find average)
        start = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0 ; i < 10 ; i++)
        {
            network.feed_forward(x, out);
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_ff10 = end - start;
        double t_ff = t_ff10.count() / 10.0;

        // Timing for back-propagation (repeated 10 times, find average)
        start = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0 ; i < 10 ; i++)
        {
            network.compute_gradients_backprop(x, y, dW, db);
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_bp10 = end - start;
        double t_bp = t_bp10.count() / 10.0;

        // Timing for finite differencing (not repeated due to long run time)
        start = std::chrono::high_resolution_clock::now();
        network.compute_gradients_finite_difference(x, y, dW, db);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_fd1 = end - start;
        double t_fd = t_fd1.count();

        // Write results to file
        outfile << N_layer << " " << N_neuron << " " << t_ff << " " << t_bp << " "  << t_fd << std::endl;

    }

    outfile.close();

    return 0;
}
