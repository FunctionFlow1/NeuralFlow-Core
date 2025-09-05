// core_engine.cpp - NeuralFlow-Core's high-performance C++ backend
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

namespace NeuralFlow {

// Simple Matrix class for basic linear algebra operations
class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("Matrix dimensions mismatch for multiplication");
        }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Element-wise addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions mismatch for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Apply a function element-wise (e.g., activation functions)
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    // Print matrix (for debugging)
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Basic ReLU activation function
double relu(double x) {
    return std::max(0.0, x);
}

// Forward pass for a simple dense layer
Matrix dense_layer_forward(const Matrix& input, const Matrix& weights, const Matrix& bias) {
    Matrix output = input * weights;
    return output + bias;
}

} // namespace NeuralFlow

// Example usage (can be compiled as a shared library or executable)
int main() {
    NeuralFlow::Matrix input(1, 3);
    input.data = {{1.0, 2.0, 3.0}};

    NeuralFlow::Matrix weights(3, 2);
    weights.data = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};

    NeuralFlow::Matrix bias(1, 2);
    bias.data = {{0.0, 0.0}};

    NeuralFlow::Matrix output = NeuralFlow::dense_layer_forward(input, weights, bias);
    std::cout << "Dense Layer Output:" << std::endl;
    output.print();

    NeuralFlow::Matrix activated_output = output.apply(NeuralFlow::relu);
    std::cout << "ReLU Activated Output:" << std::endl;
    activated_output.print();

    return 0;
}
