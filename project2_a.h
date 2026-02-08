#ifndef PROJECT2_A_H
#define PROJECT2_A_H
#include "project2_a_basics.h"

namespace BasicDenseLinearAlgebra
{

// NeuralNetworkLayer class represents a single non-input layer
// of the network. It stores all data asscociated with that layer.
class NeuralNetworkLayer
{

public:

    // Constructor for a layer, given:
    // n_input (number of inputs from previous layer), 
    // n_output (number of neurons in this layer),
    // and pointer to activation function used by all neurons in 
    // this layer.
    NeuralNetworkLayer(const unsigned& n_input,
                       const unsigned& n_output,
                       ActivationFunction* activation_function_pt)
    :
    // Store input/output sizes
    N_input(n_input),
    N_output(n_output),
    // Construct weight matrix with given dimensions
    Weight_matrix(n_output, n_input),
    // Construct bias vector with given length
    Bias_vector(n_output),
    // Store pointer to activation function
    Activation_function_pt(activation_function_pt)
    {

    }

    // Number of inputs into this layer
    unsigned n_input() const
    {
        return N_input;
    }

    // Number of neurons in this layer
    unsigned n_output() const
    {
        return N_output;
    }

    // Access to weight matrix (can be changed/non-const)
    DoubleMatrix& weight_matrix()
    {
        return Weight_matrix;
    }

    // Access to weight matrix (cannot be changed/const)
    const DoubleMatrix& weight_matrix() const
    {
        return Weight_matrix;
    }

    // Access to bias vector (cannot be changed/non-const)
    DoubleVector& bias_vector()
    {
        return Bias_vector;
    }

    // Access to bias vector (can be changed/const)
    const DoubleVector& bias_vector() const
    {
        return Bias_vector;
    }

    // Access to the activation function pointer
    ActivationFunction* activation_function_pt() const
    {
        return Activation_function_pt;
    }



    // Function to perform forward pass through a layer, given
    // a_prev (input from previous layer) and 
    // returning a_curr (output from current layer).
    void forward(const DoubleVector& a_prev,
                 DoubleVector& a_curr) const
    {
        // Input size must match expected number of inputs
        if (a_prev.n() != N_input)
        {
            throw std::runtime_error(
                "NeuralNetworkLayer::forward: input size does not match N_input.");
        }

        // Resize output vector to correct length
        a_curr.resize(N_output);

        // For each neuron i in this layer:
        for (unsigned i = 0; i < N_output; i++)
        {
            // Define bias vector for neuron i
            double z_i = Bias_vector[i];

            // Add contribution from ALL inputs j
            for (unsigned j = 0; j < N_input; j++)
            {
                z_i += Weight_matrix(i, j) * a_prev[j];
            }

            // Apply activation function
            a_curr[i] = (*Activation_function_pt).sigma(z_i);
        }
    }


private:

    unsigned N_input;
    unsigned N_output;
    DoubleMatrix Weight_matrix;
    DoubleVector Bias_vector;
    ActivationFunction* Activation_function_pt;

};




// NeuralNetwork class represents the entire neural network,
// derived from NeuralNetworkBasis.
class NeuralNetwork : public NeuralNetworkBasis
{

public:

    // Constructor that passes the number of inputs, 
    // n_input, and a vector of pairs which for the
    // lth non-input layer, contains:
    // - the number of neurons in the layer
    // (given by non_input_layer[l].first)
    // - a pointer to the activation function to be used
    //      by all neurons in that layer
    // (given by non_input_layer[l].second)
    // 
    // Note: l=0,1,.. with l=0 being the first non-input layer
    NeuralNetwork(const unsigned& n_input,
                  const std::vector<std::pair<unsigned,ActivationFunction*>>& non_input_layer)
    :
    N_input(n_input)
    {
        // Assume the first non-input layer receives n_input
        // values from the input layer.
        unsigned n_prev = n_input;

        // Loop over every non-input layer specification
        for (unsigned l = 0; l < non_input_layer.size(); l++)
        {
            // Number of neurons in this layer
            unsigned n_current = non_input_layer[l].first;

            // Pointer to activation function for this layer
            ActivationFunction* f_pt = non_input_layer[l].second;

            // Build NeuralNetworkLayer and add to end of 
            // Layers vector
            Layers.push_back(NeuralNetworkLayer(n_prev, n_current, f_pt));

            // Redefine the number of inputs for next layer to the
            // number of outputs of the current layer
            n_prev = n_current;
        }

    }



    // Implementation of feed_forward function, 
    // which computes the output
    // from the final layer given the input to the network
    void feed_forward(const DoubleVector& input,
                      DoubleVector& output) const override
    {
        // Input dimension must match N_input
        if (input.n() != N_input)
        {
            throw std::runtime_error(
                "NeuralNetwork::feed_forward: input size does not match N_input.");
        }

        // We must have at least one non-input layer
        if (Layers.size() == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::feed_forward: network has no non-input layers.");
        }

        // a_current stores the activations of the "current" layer
        DoubleVector a_current(input);

        // For each non-input layer, compute the new activations
        for (unsigned l = 0; l < Layers.size(); l++)
        {
            DoubleVector a_next;
            Layers[l].forward(a_current, a_next);
            // The output of this layer becomes the input to the next
            a_current = a_next;
        }

        // output = a[L] = a_current once passed through all layers
        output = a_current;
    }



    // Implementation of cost function, which computes the cost for
    // a single input and corresponding single target output.
    double cost(const DoubleVector& input,
                const DoubleVector& target_output) const override
    {
        // Run network to get output for given input
        DoubleVector network_output;
        feed_forward(input, network_output);

        // Check that target dimension matches network output dimension
        if (network_output.n() != target_output.n())
        {
            throw std::runtime_error(
                "NeuralNetwork::cost: size of target_output does not match size of network output.");
        }

        // Initialize sum of squares
        double sum_sq = 0.0;
        // Compute sum of squared differences between output and target
        for (unsigned j = 0; j < network_output.n(); j++)
        {
            double diff = network_output[j] - target_output[j];
            sum_sq += diff * diff;
        }

        // Return the cost
        double C_x = 0.5 * sum_sq;
        return C_x;
    }



    // Implementation of cost_for_training_data function, which 
    // computes the total cost for a given set of training data
    // (N inputs and associated target outputs).
    double cost_for_training_data(
        const std::vector< std::pair<DoubleVector,DoubleVector> > training_data) 
        const override
    {
        // Training data must not be empty
        if (training_data.size() == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::cost_for_training_data: training_data is empty.");
        }

        // Initialize total cost
        double total_cost = 0.0;

        // Number of training data points given
        unsigned N = training_data.size();

        // Loop over training data
        for (unsigned i = 0; i < N; i++)
        {
            // Extract input and target for ith training example
            const DoubleVector& x_i = training_data[i].first;
            const DoubleVector& y_i = training_data[i].second;

            // Use cost function to get cost for this example
            double C_i = cost(x_i, y_i);

            // Add to running total
            total_cost += C_i;
        }

        // Compute average cost over all training examples and return
        double N_double = N;
        double C = total_cost / N_double;
        return C;
    }



    // Implementation of write_parameters_to_disk function, which
    // writes all network data to a given file.
    void write_parameters_to_disk(const std::string& filename) 
        const override
    {
        // Open output file, throw error if it fails
        std::ofstream outfile(filename.c_str());
        if (!outfile)
        {
            throw std::runtime_error(
                "NeuralNetwork::write_parameters_to_disk: cannot open file.");
        }

        // Loop over all non-input layers
        for (unsigned l = 0; l < Layers.size(); l++)
        {
            const NeuralNetworkLayer& layer = Layers[l];

            // Output activation function name
            ActivationFunction* act = layer.activation_function_pt();
            std::string act_name = (*act).name();
            outfile << act_name << std::endl;

            // Output number of inputs m
            unsigned m = layer.n_input();
            outfile << m << std::endl;

            // Output number of neurons n
            unsigned n = layer.n_output();
            outfile << n << std::endl;

            // Output bias vector entries
            const DoubleVector& b = layer.bias_vector();
            b.output(outfile);

            // Output weight matrix entries
            const DoubleMatrix& W = layer.weight_matrix();
            W.output(outfile);

        }

        outfile.close();
    }



    // Implementation of read_parameters_from_disk function, which
    // reads all network data from a given file.
    void read_parameters_from_disk(const std::string& filename) override
    {
        // Open input file, throw error if it fails
        std::ifstream infile(filename.c_str());
        if (!infile)
        {
            throw std::runtime_error(
                "NeuralNetwork::read_parameters_from_disk: cannot open file.");
        }

        // Loop over all non-input layers
        for (unsigned l = 0; l < Layers.size(); l++)
        {
            NeuralNetworkLayer& layer = Layers[l];

            // Read activation function name from file
            std::string file_act_name;
            infile >> file_act_name;
            if (!infile)
            {
                throw std::runtime_error(
                    "NeuralNetwork::read_parameters_from_disk: failed "
                    "to read activation function name.");
            }

            // Our expected activation function name for this layer,
            // throw error if it doesn't match what we read from file
            ActivationFunction* act = layer.activation_function_pt();
            std::string expected_act_name = (*act).name();
            if (file_act_name != expected_act_name)
            {
                throw std::runtime_error("NeuralNetwork::read_parameters_from_disk: "
                    "activation function name in file does not match "
                    "the activation function used in this network.");
            }

            // Read input dimension m from file, throw error if not 
            // read correctly or doesn't match our structure
            unsigned m_file = 0;
            infile >> m_file;
            if (!infile)
            {
                throw std::runtime_error(
                    "NeuralNetwork::read_parameters_from_disk: failed "
                    "to read input dimension m.");
            }
            unsigned m_expected = layer.n_input();
            if (m_file != m_expected)
            {
                throw std::runtime_error("NeuralNetwork::read_parameters_from_disk: "
                    "input dimension m read from file does not match "
                    "the network structure.");
            }

            // Read number of neurons n from file, throw error if not
            // read correctly or doesn't match our structure
            unsigned n_file = 0;
            infile >> n_file;
            if (!infile)
            {
                throw std::runtime_error(
                    "NeuralNetwork::read_parameters_from_disk: failed "
                    "to read number of neurons n.");
            }
            unsigned n_expected = layer.n_output();
            if (n_file != n_expected)
            {
                throw std::runtime_error("NeuralNetwork::read_parameters_from_disk: "
                    "number of neurons n read from file does not match "
                    "the network structure.");
            }

            // Read bias vector b from file, throw error if not read
            // correctly
            DoubleVector& b = layer.bias_vector();
            b.read(infile);
            if (!infile)
            {
                throw std::runtime_error(
                    "NeuralNetwork::read_parameters_from_disk: error "
                    "while reading bias vector.");
            }

            // Read weight matrix W from file, throw error if not read
            // correctly
            DoubleMatrix& W = layer.weight_matrix();
            W.read(infile);
            if (!infile)
            {
                throw std::runtime_error(
                    "NeuralNetwork::read_parameters_from_disk: error "
                    "while reading weight matrix.");
            }
        }

        infile.close();
    }



    // Implementation of train function, which trains the network
    // using stochastic gradient descent, given training data,
    // the learning rate, target cost and the maximum number of
    // iterations. 
    //
    // A file name for convergence history can optionally be provided.
    void train(
        const std::vector< std::pair<DoubleVector,DoubleVector> >& training_data,
        const double& learning_rate,
        const double& tol_training,
        const unsigned& max_iter,
        const std::string& convergence_history_file_name = "") override
    {

        // Number of training data points must be more than zero
        const unsigned N_training = training_data.size();
        if (N_training == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::train: No training data provided.");
        }

        // Network must have at least one non-input layer
        if (Layers.size() == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::train: Network has no non-input layers.");
        }

        // Open file for convergence history if requested
        // Throw error if file cannot be opened
        std::ofstream history_file;
        if (convergence_history_file_name != "")
        {
            history_file.open(convergence_history_file_name.c_str());
            if (!history_file)
            {
                throw std::runtime_error(
                    "NeuralNetwork::train: Could not open convergence "
                    "history file.");
            }
        }

        // Compare gradients from back-propagation and finite-difference
        // for the first training data point
        {
            // Extract first training data point
            const DoubleVector& x0 = training_data[0].first;
            const DoubleVector& y0 = training_data[0].second;

            // Define vectors of derivatives computed by the two methods
            std::vector<DoubleMatrix> dW_fd;
            std::vector<DoubleVector> db_fd;
            std::vector<DoubleMatrix> dW_bp;
            std::vector<DoubleVector> db_bp;

            // Back-propagation gradients
            compute_gradients_backprop(x0, y0, dW_bp, db_bp);

            // Finite-difference gradients
            compute_gradients_finite_difference(x0, y0, dW_fd, db_fd);

            // Compare the two sets of derivatives and print maximum difference
            double max_diff = 0.0;
            const unsigned n_layers_local = Layers.size();
            for (unsigned l = 0; l < n_layers_local; l++)
            {
                unsigned n_out = Layers[l].n_output();
                unsigned n_in  = Layers[l].n_input();

                // Compare weight derivatives
                for (unsigned j = 0; j < n_out; j++)
                {
                    for (unsigned k = 0; k < n_in; k++)
                    {
                        double diff = dW_fd[l](j,k) - dW_bp[l](j,k);
                        if (diff < 0.0) 
                        {
                            diff = -diff;
                        }
                        if (diff > max_diff)
                        { 
                            max_diff = diff;
                        }
                    }
                }

                // Compare bias derivatives
                for (unsigned j = 0; j < n_out; j++)
                {
                    double diff = db_fd[l][j] - db_bp[l][j];
                    if (diff < 0.0) 
                    {
                        diff = -diff;
                    }
                    if (diff > max_diff) 
                    {
                        max_diff = diff;
                    }
                }
            }

            std::cout << "Maximum difference between finite-difference "
                      << "and back-propagation gradients = "
                      << max_diff << std::endl;
        }


        /// Stochastic gradient descent algorithm

        // Random integer in [0, N_training-1] to choose a data point
        std::uniform_int_distribution<unsigned> random_index(
            0, N_training - 1);

        // Compute initial total cost (convergence check)
        double total_cost = cost_for_training_data(training_data);

        // Write first entry in convergence history
        unsigned iter = 0;
        if (history_file.is_open())
        {
            history_file << iter << " " << total_cost << std::endl;
        }

        // Define how often we check convergence
        const unsigned check_interval = 500;

        // Keep updating weights and biases until either the cost is 
        // low enough or we hit max_iter.
        while ((iter < max_iter) && (total_cost > tol_training))
        {
            // Increase iteration count
            iter++;

            // Choose random training data point
            unsigned i = random_index(
                RandomNumber::Random_number_generator);
            const DoubleVector& x_i = training_data[i].first;
            const DoubleVector& y_i = training_data[i].second;

            // Compute gradients using back-propagation
            std::vector<DoubleMatrix> dW;
            std::vector<DoubleVector> db;
            compute_gradients_backprop(x_i, y_i, dW, db);

            // Update all weights and biases using the computed 
            // gradients
            const unsigned n_layers_local = Layers.size();
            for (unsigned l = 0; l < n_layers_local; l++)
            {
                DoubleMatrix& W = Layers[l].weight_matrix();
                DoubleVector& b = Layers[l].bias_vector();

                unsigned n_out = Layers[l].n_output();
                unsigned n_in  = Layers[l].n_input();

                for (unsigned j = 0; j < n_out; j++)
                {
                    // Update bias entry
                    b[j] -= learning_rate * db[l][j];

                    // Update all weights into neuron j
                    for (unsigned k = 0; k < n_in; k++)
                    {
                        W(j,k) -= learning_rate * dW[l](j,k);
                    }
                }
            }

            // Recompute total cost every 'check_interval' iterations
            // and on final iteration (if 'max_iter' reached)
            // and write to convergence history file (if given)
            if ((iter % check_interval == 0) || (iter == max_iter))
            {
                total_cost = cost_for_training_data(training_data);

                if (history_file.is_open())
                {
                    history_file << iter << " " << total_cost << std::endl;
                }
            }
        }

        // Close convergence history file (if given)
        if (history_file.is_open())
        {
            history_file.close();
        }

        // Message indicating why iteration stopped
        if (total_cost <= tol_training)
        {
            std::cout << "Training finished: cost "
                      << total_cost
                      << " <= tolerance " << tol_training
                      << std::endl;
        }
        else
        {
            std::cout << "Training stopped: reached max_iter = "
                      << max_iter << ", final cost = "
                      << total_cost << std::endl;
        }
    }





    // train_interlaced function, which has identical
    // purpose to previously given train function,
    // but combines computation of derivatives of the cost
    // and the updates of the network's weights and biases
    // for efficiency and a reduction in storage required
    void train_interlaced(
                    const std::vector< std::pair<DoubleVector,DoubleVector> >& training_data,
                    const double& learning_rate,
                    const double& tol_training,
                    const unsigned& max_iter,
                    const std::string& convergence_history_file_name = "")
    {

        // Number of training data points must be more than zero
        const unsigned N_training = training_data.size();
        if (N_training == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::train_interlaced: No training data provided.");
        }

        // Network must have at least one non-input layer
        if (Layers.size() == 0)
        {
            throw std::runtime_error(
                "NeuralNetwork::train_interlaced: Network has no non-input layers.");
        }

        // Open file for convergence history if requested
        // Throw error if file cannot be opened
        std::ofstream history_file;
        if (convergence_history_file_name != "")
        {
            history_file.open(convergence_history_file_name.c_str());
            if (!history_file)
            {
                throw std::runtime_error(
                    "NeuralNetwork::train_interlaced: Could not open convergence "
                    "history file.");
            }
        }

        /// Stochastic gradient descent algorithm

        // Random integer in [0, N_training-1] to choose a data point
        std::uniform_int_distribution<unsigned> random_index(
            0, N_training - 1);

        // Compute initial total cost
        double total_cost = cost_for_training_data(training_data);

        // Write first entry in convergence history
        unsigned iter = 0;
        if (history_file.is_open())
        {
            history_file << iter << " " << total_cost << std::endl;
        }

        // Define how often we check convergence
        const unsigned check_interval = 500;

        // Keep updating weights and biases until either the cost is
        // low enough or we hit max_iter.
        while ((iter < max_iter) && (total_cost > tol_training))
        {
            // Increase iteration counter
            iter++;

            // Choose random training data point
            unsigned i = random_index(RandomNumber::Random_number_generator);

            // Input vector x and target output y for random point
            const DoubleVector& x_i = training_data[i].first;
            const DoubleVector& y_i = training_data[i].second;

            /// INTERLACED BACK-PROPAGATION
            /// compute local derivatives to upate weight and bias
            /// immediately rather than storing them for all layers

            // Forward pass to store activations a and inputs z
            const unsigned n_layers = Layers.size();
            std::vector<DoubleVector> a(n_layers + 1);
            std::vector<DoubleVector> z(n_layers);

            // Set input layer activation a[0]
            a[0].resize(N_input);
            for (unsigned j = 0; j < N_input; j++)
            {
                a[0][j] = x_i[j];
            }

            // Feed-forward through each layer (to find z[l], a[l+1])
            for (unsigned l = 0; l < n_layers; l++)
            {
                // Access weights, biases, and activation function 
                // for current layer
                DoubleMatrix& W = Layers[l].weight_matrix();
                DoubleVector& b = Layers[l].bias_vector();
                ActivationFunction* act = Layers[l].activation_function_pt();

                unsigned n_out = Layers[l].n_output();
                unsigned n_in  = Layers[l].n_input();

                // Allocate space for z[l] and a[l+1]
                z[l].resize(n_out);
                a[l+1].resize(n_out);

                // Compute z[l] and a[l+1]
                for (unsigned j = 0; j < n_out; j++)
                {
                    double s = b[j];
                    for (unsigned k = 0; k < n_in; k++)
                    {
                        s += W(j,k) * a[l][k];
                    }
                    z[l][j] = s;
                    a[l+1][j] = (*act).sigma(s);
                }
            }

            // Backward pass for outer layer

            // delta represents dC/dz
            // delta_curr for now is dC/dz at output
            DoubleVector delta_curr;

            {
                const unsigned l = n_layers - 1;

                ActivationFunction* act = Layers[l].activation_function_pt();

                // resize delta to correct size
                unsigned n_out = Layers[l].n_output();
                delta_curr.resize(n_out);

                // Compute delta at output
                for (unsigned j = 0; j < n_out; j++)
                {
                    // Error in output neuron j
                    const double diff = a[n_layers][j] - y_i[j];

                    // Compute current delta jth component
                    delta_curr[j] = diff * (*act).dsigma(z[l][j]);
                }
            }

            // Backward pass through all non-input layers
            // (now that we have access to a 'previous' delta
            // for the outer layer)
            for (int l_int = static_cast<int>(n_layers) - 1; l_int >= 0; l_int--)
            {
                unsigned l = static_cast<unsigned>(l_int);

                // Access current layer parameters
                DoubleMatrix& W = Layers[l].weight_matrix();
                DoubleVector& b = Layers[l].bias_vector();

                unsigned n_out = Layers[l].n_output();
                unsigned n_in  = Layers[l].n_input();

                // Compute delta for previous layer
                DoubleVector delta_prev;

                // if current layer is not the first layer
                if (l > 0)
                {
                    // Define activation function for previous layer
                    ActivationFunction* act_prev = Layers[l-1].activation_function_pt();

                    // Resize previous delta to appropriate size
                    unsigned n_prev = Layers[l-1].n_output();
                    delta_prev.resize(n_prev);

                    // Use formulae given to compute delta for previous layer
                    for (unsigned j = 0; j < n_prev; j++)
                    {
                        double sum = 0.0;

                        // Weighted sum of errors from next layer
                        for (unsigned k = 0; k < n_out; k++)
                        {
                            sum += W(k,j) * delta_curr[k];
                        }

                        delta_prev[j] = (*act_prev).dsigma(z[l-1][j]) * sum;
                    }
                }

                // Update weights and biases for layer l (SGD step)
                for (unsigned j = 0; j < n_out; j++)
                {
                    // Update bias
                    b[j] -= learning_rate * delta_curr[j];

                    // Update weights
                    for (unsigned k = 0; k < n_in; k++)
                    {
                        W(j,k) -= learning_rate * (delta_curr[j] * a[l][k]);
                    }
                }

                // Previous delta becomes current delta for next iteration
                if (l > 0)
                {
                    delta_curr = delta_prev;
                }
            }

            // Recompute total cost every 'check_interval' iterations
            // and on final iteration (if 'max_iter' reached)
            // and write to convergence history file (if given)
            if ((iter % check_interval == 0) || (iter == max_iter))
            {
                total_cost = cost_for_training_data(training_data);

                if (history_file.is_open())
                {
                    history_file << iter << " " << total_cost << std::endl;
                }
            }
        }


        // Close convergence history file if opened
        if (history_file.is_open())
        {
            history_file.close();
        }

        // Message indicating why iteration stopped
        if (total_cost <= tol_training)
        {
            std::cout << "Training finished: cost "
                      << total_cost
                      << " <= tolerance " << tol_training
                      << std::endl;
        }
        else
        {
            std::cout << "Training stopped: reached max_iter = "
                      << max_iter << ", final cost = "
                      << total_cost << std::endl;
        }

    }





    // Implementation of initialise_parameters function, which
    // initialises all weights and biases randomly from a normal
    // distribution with given mean and standard deviation.
    void initialise_parameters(const double& mean,
                               const double& std_dev) override
    {
        // Define normal distribution with given mean and std_dev
        std::normal_distribution<> normal_dist(mean, std_dev);

        // Loop over all layers
        for (unsigned l = 0; l < Layers.size(); l++)
        {
            // Current layer
            NeuralNetworkLayer& layer = Layers[l];

            // Randomise biases in this layer
            DoubleVector& b = layer.bias_vector();
            for (unsigned j = 0; j < b.n(); j++)
            {
                // Store one random number per bias entry
                b[j] = normal_dist(RandomNumber::Random_number_generator);
            }

            // Randomise weights in this layer
            DoubleMatrix& W = layer.weight_matrix();
            // Number of output neurons (rows) and inputs (columns)
            unsigned n_rows = W.n();
            unsigned n_cols = W.m();

            // Loop over all weight entries
            for (unsigned i = 0; i < n_rows; i++)
            {
                for (unsigned j = 0; j < n_cols; j++)
                {
                    // A random number per weight entry
                    W(i,j) = normal_dist(RandomNumber::Random_number_generator);
                }
            }
        }
    }



    // Implementation of initialise_parameters_for_test function,
    // (used for testing)
    // which sets all weights and biases to a specific value.
    void initialise_parameters_for_test()
    {
        // Loop over all layers
        for (unsigned l = 0; l < Layers.size(); l++)
        {
            NeuralNetworkLayer& layer = Layers[l];

            // Define biases
            DoubleVector& b = layer.bias_vector();
            for (unsigned j = 0; j < b.n(); j++)
            {
                b[j] = l + (j*0.01);
            }

            // Define weights
            DoubleMatrix& W = layer.weight_matrix();
            for (unsigned i = 0; i < W.n(); i++)
            {
                for (unsigned j = 0; j < W.m(); j++)
                {
                    W(i,j) = (0.1*(l + 1)) + (0.01*i) + (0.001*j);
                }
            }
        }
    }



    // Function to return number of non-input layers in the network
    unsigned n_layers() const
    {
        return Layers.size();
    }


private:

    unsigned N_input;
    std::vector<NeuralNetworkLayer> Layers;



    // Function to compute gradients dW and db 
    // using back-propagation algorithm given training pair
    void compute_gradients_backprop(
        const DoubleVector& input,
        const DoubleVector& target_output,
        std::vector<DoubleMatrix>& dW,
        std::vector<DoubleVector>& db) const
    {
        // Number of non-input layers
        const unsigned L = Layers.size();

        // Store all activations and weighted inputs
        std::vector<DoubleVector> a(L + 1);
        std::vector<DoubleVector> z(L);

        // a[0] = input
        a[0].resize(N_input);
        for (unsigned i = 0; i < N_input; i++)
        {
            a[0][i] = input[i];
        }

        // Forward pass
        // Loop over all layers to store a[l+1] and z[l]
        for (unsigned l = 0; l < L; l++)
        {
            // Current layer variables
            const NeuralNetworkLayer& layer = Layers[l];
            unsigned n_in  = layer.n_input();
            unsigned n_out = layer.n_output();

            // Access weight matrix, bias vector and activation function
            const DoubleMatrix& W = layer.weight_matrix();
            const DoubleVector& b = layer.bias_vector();
            ActivationFunction* act = layer.activation_function_pt();

            // Resize z[l] and a[l+1]
            z[l].resize(n_out);
            a[l+1].resize(n_out);

            // For each neuron j in current layer
            for (unsigned j = 0; j < n_out; j++)
            {
                // Compute weighted input z[l][j]
                double z_j = b[j];
                // Add contribution from all inputs k
                for (unsigned k = 0; k < n_in; k++)
                {
                    z_j += W(j,k) * a[l][k];
                }
                z[l][j] = z_j;

                // Apply activation function to obtain next activation
                a[l+1][j] = (*act).sigma(z_j);
            }
        }

        // Store vector for delta dC/dz
        std::vector<DoubleVector> delta(L);

        // Backward pass to find deltas
        // Last layer: l = L-1
        unsigned l = L - 1;
        const NeuralNetworkLayer& layer = Layers[l];
        unsigned n_out = layer.n_output();
        ActivationFunction* act = layer.activation_function_pt();

        // Resize delta[l]
        delta[l].resize(n_out);

        // For each neuron j in last layer
        for (unsigned j = 0; j < n_out; j++)
        {
            // Compute delta[L][j] by Haramard product
            double diff = a[L][j] - target_output[j];
            double d_sigma = (*act).dsigma(z[l][j]);
            delta[l][j] = diff * d_sigma;
        }

        // Previous layers
        for (int l_int = static_cast<int>(L) - 2; l_int >= 0; l_int--)
        {
            unsigned l = static_cast<unsigned>(l_int);

            // Define variables for current and next layer
            const NeuralNetworkLayer& current_layer = Layers[l];
            const NeuralNetworkLayer& next_layer    = Layers[l+1];
            unsigned n_out_current = current_layer.n_output();
            unsigned n_out_next    = next_layer.n_output();
            unsigned n_in_current  = current_layer.n_input();
            ActivationFunction* act_current =
                current_layer.activation_function_pt();

            // Access weight matrix of next layer
            const DoubleMatrix& W_next = next_layer.weight_matrix();

            // Resize delta[l]
            delta[l].resize(n_out_current);

            // For each neuron j in current layer
            for (unsigned j = 0; j < n_out_current; j++)
            {
                // Compute delta for current layer by back-propagation
                double sum = 0.0;
                for (unsigned k = 0; k < n_out_next; k++)
                {
                    sum += W_next(k,j) * delta[l+1][k];
                }

                // Multiply by derivative of activation function
                // from Haramard-free equivalent
                double d_sigma = (*act_current).dsigma(z[l][j]);
                delta[l][j] = d_sigma * sum;
            }
        }

        // Using computed deltas, prepare gradients dW and db
        dW.clear();
        db.clear();
        dW.reserve(L);
        db.reserve(L);

        // Loop over all layers
        for (unsigned l = 0; l < L; l++)
        {
            // Define current layer
            const NeuralNetworkLayer& layer = Layers[l];
            unsigned n_in  = layer.n_input();
            unsigned n_out = layer.n_output();

            // Prepare matrix/vector for gradients
            DoubleMatrix dW_layer(n_out, n_in);
            DoubleVector db_layer(n_out);

            // For each neuron j in current layer
            for (unsigned j = 0; j < n_out; j++)
            {
                // Bias derivative
                db_layer[j] = delta[l][j];
                // Weight derivatives
                for (unsigned k = 0; k < n_in; k++)
                {
                    dW_layer(j,k) = delta[l][j] * a[l][k];
                }
            }

            // Store gradients for this layer
            dW.push_back(dW_layer);
            db.push_back(db_layer);
        }
    }



    // Function to compute gradients dW and db 
    // using finite difference algorithm given training pair
    void compute_gradients_finite_difference(
        const DoubleVector& input,
        const DoubleVector& target_output,
        std::vector<DoubleMatrix>& dW,
        std::vector<DoubleVector>& db)
    {
        // Number of non-input layers
        const unsigned L = Layers.size();

        // Step size for finite differencing
        const double h = 1.0e-6;

        // Prepare output vectors/matrices
        dW.clear();
        db.clear();
        dW.reserve(L);
        db.reserve(L);

        // Loop over all layers
        for (unsigned l = 0; l < L; l++)
        {
            // Define variables for current layer
            NeuralNetworkLayer& layer = Layers[l];

            unsigned n_in  = layer.n_input();
            unsigned n_out = layer.n_output();

            // Store gradient matrix/vector of appropriate size
            DoubleMatrix dW_layer(n_out, n_in);
            DoubleVector db_layer(n_out);

            // Parameters for finite difference
            DoubleMatrix& W = layer.weight_matrix();
            DoubleVector& b = layer.bias_vector();

            /// Finite-difference derivatives

            // Loop over all weights
            for (unsigned j = 0; j < n_out; j++)
            {
                for (unsigned k = 0; k < n_in; k++)
                {
                    // Store original weight
                    double original = W(j,k);

                    // Cost at w + h
                    W(j,k) = original + h;
                    double C_plus = cost(input, target_output);

                    // Cost at w - h
                    W(j,k) = original - h;
                    double C_minus = cost(input, target_output);

                    // Restore original weight (so network is unchanged)
                    W(j,k) = original;

                    // Central difference approximation
                    dW_layer(j,k) = (C_plus - C_minus) / (2.0 * h);
                }
            }

            // Loop over all biases
            for (unsigned j = 0; j < n_out; j++)
            {
                // Store original bias
                double original = b[j];

                // Cost at b + h
                b[j] = original + h;
                double C_plus = cost(input, target_output);

                // Cost at b - h
                b[j] = original - h;
                double C_minus = cost(input, target_output);

                // Restore original bias (so network is unchanged)
                b[j] = original;

                // Central difference approximation
                db_layer[j] = (C_plus - C_minus) / (2.0 * h);
            }

            // Store gradients for this layer
            dW.push_back(dW_layer);
            db.push_back(db_layer);
        }
    }

};

}

#endif