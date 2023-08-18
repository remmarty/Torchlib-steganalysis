#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <iostream>

const char* dataRoot = "./data";
const int64_t numberOfEpochs = 10;
const int64_t trainBatchSize = 16;
const int64_t testBatchSize = 64;

struct Net : torch::nn::Module {
	torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5;
	torch::nn::BatchNorm2d batchNorm30, batchNorm32, batchNorm64, batchNorm128;
	torch::nn::AvgPool2d avgPool;
	torch::nn::MaxPool2d maxPool;
	torch::nn::Linear fc1, fc2, fc3;

	Net()
		: conv1(register_conv("conv1", 30, 30, 5)),
		conv2(register_conv("conv2", 30, 30, 5)),
		conv3(register_conv("conv3", 30, 32, 3)),
		conv4(register_conv("conv4", 32, 64, 3)),
		conv5(register_conv("conv5", 64, 128, 3)),
		batchNorm30(register_batch_norm("batchNorm30", 30)),
		batchNorm32(register_batch_norm("batchNorm32", 32)),
		batchNorm64(register_batch_norm("batchNorm64", 64)),
		batchNorm128(register_batch_norm("batchNorm128", 128)),
		avgPool(torch::nn::AvgPool2dOptions(5).stride(2).padding(2)),
		maxPool(torch::nn::MaxPool2dOptions(3)),
		fc1(register_fc("fc1", 128, 256)),
		fc2(register_fc("fc2", 256, 1024)),
		fc3(register_fc("fc3", 1024, 2))
	{
	}
	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(batchNorm30->forward(conv1->forward(x)));
		x = avgPool->forward(torch::relu(batchNorm30->forward(conv2->forward(x))));
		x = avgPool->forward(torch::relu(batchNorm32->forward(conv3->forward(x))));
		x = avgPool->forward(torch::relu(batchNorm64->forward(conv4->forward(x))));
		x = maxPool->forward(torch::relu(batchNorm128->forward(conv5->forward(x))));

		x = x.view({ -1, 128 });
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), 1);
		return x;
	}

private:
	torch::nn::Conv2d register_conv(const std::string& name, int in_channels, int out_channels, int kernel_size) {
		return register_module(name, torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)));
	}

	torch::nn::BatchNorm2d register_batch_norm(const std::string& name, int num_features) {
		return register_module(name, torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features)));
	}

	torch::nn::Linear register_fc(const std::string& name, int in_features, int out_features) {
		return register_module(name, torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features)));
	}
};

void check_cuda() {
	if (torch::cuda::is_available())
		std::cout << "CUDA is available! Training on GPU" << std::endl;
	else
		std::cout << "CUDA is not available! Training on CPU" << std::endl;
}

void train() {
	for (size_t epoch = 1; epoch <= numberOfEpochs; ++epoch) {

	}
}

void test() {

}

int main() {
	check_cuda();
	auto net = std::make_shared<Net>();
}

