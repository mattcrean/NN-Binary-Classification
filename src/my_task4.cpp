#include "project2_a.h"
#include <chrono>
#include <iostream>
#include <iomanip>

// Driver code for Analysis Task 4
//
// Calculates the time taken for two networks with
// the same parameters to train under the original
// training process and the interlaced training
// process and outputs the ratio of the two times
// to a file.

using namespace BasicDenseLinearAlgebra;

int main()
{
    // Open file to write ratios to
    std::ofstream outfile("task4_ratio.dat");

    // Define timer
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    std::vector<std::pair<DoubleVector,DoubleVector>> training_data;

    unsigned n_input = 2;

    // Loop for several widths of neural network
    for (int n = 2; n < 41; n+=2)
    {
        ActivationFunction* tanh = new TanhActivationFunction;

        // Define layers
        std::vector< std::pair<unsigned, ActivationFunction*> > non_input_layer;
        non_input_layer.push_back(std::make_pair(n, tanh));
        non_input_layer.push_back(std::make_pair(n, tanh));
        non_input_layer.push_back(std::make_pair(1, tanh));

        NeuralNetwork net_train(n_input, non_input_layer);
        NeuralNetwork net_train_interlaced(n_input, non_input_layer);

        // Train networks
        net_train.read_training_data("project_training_data.dat", training_data);

        // Initialise parameters by given normal distribution
        net_train.initialise_parameters(0.0, 0.1);
        const double eta = 0.1;
        const double tol_training = 1e-4;
        const unsigned max_iter = 100000;

        // Define same initial parameters for fair comparison of training times
        net_train.write_parameters_to_disk("task4_initial_params.dat");
        net_train_interlaced.read_parameters_from_disk("task4_initial_params.dat");

        // Time training for original train function
        start = std::chrono::high_resolution_clock::now();
        net_train.train(
                training_data, 
                eta, 
                tol_training, 
                max_iter, 
                "task4_conv_train.dat");
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t = end - start;
        double t_train = t.count();

        // Time training for new train_interlaced function
        start = std::chrono::high_resolution_clock::now();
        net_train_interlaced.train_interlaced(
                training_data, 
                eta, 
                tol_training, 
                max_iter, 
                "task4_conv_train_interlaced.dat");
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ti = end - start;
        double t_train_inter = ti.count();

        // Output ratio of times to file
        outfile << n << " " << t_train / t_train_inter << std::endl;
    
    }

    outfile.close();

    return 0;
}
