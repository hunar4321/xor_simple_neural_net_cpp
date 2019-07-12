// @ hunar ahmad abdulrahman
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

// parameters
const float learn_rate = 0.1;
const int ins = 2;
const int nodes = 5;
const int out = 1;
const int nsamples = 4;

// data & labels
float X[nsamples][ins] = { {0, 0} ,{1, 1},{1, 0},{0, 1} };
float y[nsamples] = { 0, 0, 1, 1 };


//// some helper functions ////
// generate a random number between min and max
float gen_rand(float max, float min) {
	return ((double)rand() / (RAND_MAX + 1))* (max - min + 1) + min;
}
// helper functions to generate randomised matrices for intializing weights
std::vector<std::vector<float>> randn(int lenX, int lenY) {
	std::vector<std::vector<float>> W;
		for (int i = 0; i < lenX; i++) {
			std::vector<float> temp;
			for (int j = 0; j < lenY; j++) {
				temp.push_back(gen_rand(-1.0,1.0));
			}
			W.push_back(temp);
		}
	return W;
}
// generate zero filled vector
std::vector<float> zeros(int len) {
	std::vector<float> V;
	for (int i = 0; i < len; i++) { V.push_back(0); }
	return V;
}
// declaring and initializing some arrays
std::vector<float> ix = zeros(ins);
std::vector<float> z1 = zeros(nodes);
std::vector<float> X2 = zeros(nodes);
std::vector<float> deY1 = zeros(nodes);
std::vector<float> deX1 = zeros(nodes);
std::vector<float> dW2 = zeros(nodes);
std::vector<float> W2 = zeros(nodes);
std::vector<std::vector<float>> dW1 = randn(ins, nodes);
std::vector<std::vector<float>> W1 = randn(ins, nodes);
std::vector<float> mse;

int main() {
	// training loop
	float z2; float yhat; float deX2; float er;
	for (int i = 0; i < 100; i++) {
		float ers = 0;
		// iterate over the samples
		for (int j = 0; j < nsamples; j++) {

			// feed forward
			z2 = 0;
			ix[0] = X[j][0]; // input1
			ix[1] = X[j][1]; // input2

			for (int j = 0; j < nodes; j++) {
				z1[j] = ix[0] * W1[0][j] + ix[1] * W1[0][j];
				X2[j] = std::sin(z1[j]); // sin used as activation function (for simplicity)
				z2 += X2[0] * W2[j];
			}
			yhat = z2;

			// estimate the error
			deX2 = y[j] - yhat;
			er = deX2;

			// backpropagation of the error deX2..
			for (int j = 0; j < nodes; j++) {

				deY1[j] = deX2 * W2[j];
				deX1[j] = deY1[j] * std::cos(z1[j]); // cos is derivative of sin

				dW2[j] = deX2 * X2[j] * learn_rate;
				W2[j] = W2[j] + dW2[j]; // update weights

				for (int k = 0; k < ins; k++) {

					dW1[k][j] = deX1[j] * ix[k] * learn_rate;
					W1[k][j] = W1[k][j] + dW1[k][j]; // update weights
				}
			}
			ers = ers + std::abs(er);
		}
		mse.push_back(ers);
	}

	//print errors
	std::cout << "errors:" << std::endl;
	for (int i = 0; i < mse.size(); i++) {
		if (i % 10 == 0) {
			std::cout << mse[i] << std::endl;
		}
	}

	// compare the network predictions "yhat" to the true labels "y"
	std::cout << "------------" << std::endl;
	std::cout << "predictions:" << std::endl;
	for (int j = 0; j < nsamples; j++) {

		// feed forward using the trained weights above
		z2 = 0;
		ix[0] = X[j][0]; // input1
		ix[1] = X[j][1]; // input2

		for (int j = 0; j < nodes; j++) {
			z1[j] = ix[0] * W1[0][j] + ix[1] * W1[0][j];
			X2[j] = std::sin(z1[j]);
			z2 += X2[0] * W2[j]; 
		}
		yhat = z2;

		std::cout << " Y: " << y[j] << " yhat: " << yhat << std::endl;
	}

}
