#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <string>
#include <vector>

//  网络训练的参数
struct Options{
    int image_size=28;
    size_t train_batch_size=64;
    size_t test_batch_size=64;
    size_t iterations=10000;
    size_t log_interval=100;
    // 配置信息
    std::string datasetPath="./data/train/images";
    std::string infoFilePath="./data/train/label.txt";
    torch::DeviceType device=torch::kCPU;
};

//  静态变量,方便调用参数
static Options options;

// 定义参数键值对类型, 行数据
using Data=std::vector<std::pair<std::string, long>>;

// 自定义数据集的类
class CustomDataset: public torch::data::datasets::Dataset<CustomDataset>
{
    using Example = torch::data::Example<>;

    Data data;

    public:
        // 构造函数,传入数据集
        CustomDataset(const Data& data): data(data){}
        
        Example get(size_t index){
            std::string path = data[index].first;
            auto mat = cv::imread(path);
            assert(!mat.empty());

            // resize
            cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
            std::vector<cv::Mat> channels(1);
            cv::split(mat, channels);

            auto gray = torch::from_blob(channels[0].ptr(), {options.image_size, options.image_size}, torch::kUInt8);
            
            auto tdata = torch::cat({gray}).view({1, options.image_size, options.image_size} ).to(torch::kFloat);
            auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);

            return {tdata, tlabel};
        }

        torch::optional<size_t> size() const{
            return data.size();
        }
};


// 拿到训练数据
std::pair<Data, Data> readInfo()
{
    // 训练集,测试集
    Data train, test;
    // 打开文本文件
    std::ifstream stream(options.infoFilePath);
    assert(stream.is_open());

    long label;
    std::string path, type;

    while(true)
    {
        stream >> path >> label >> type;
        if(type == "train")
            train.push_back(std::make_pair(path, label));
        else if(type=="test")
            test.push_back(std::make_pair(path, label));
        else
            assert(false);

        if(stream.eof())
            break;
    }

    std::random_shuffle(train.begin(), train.end());
    std::random_shuffle(test.begin(), test.end());
    return std::make_pair(train, test);
}

// 定义网络
struct Network : torch::nn::Module{
    Network():
        conv1(torch::nn::Conv2dOptions(1,10,5)),
        conv2(torch::nn::Conv2dOptions(10,20,5)),
        fc1(320, 50),
        fc2(50,10)
        {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv2_drop", conv2_drop);
            register_module("fc1", fc1);
            register_module("fc2", fc2);

        }
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::FeatureDropout conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

template<typename DataLoader>
void train(Network& network, 
    DataLoader& loader, 
    torch::optim::Optimizer& optimizer, 
    size_t epoch, 
    size_t data_size)
{
    size_t index = 0;
    network.train();
    float Loss = 0, Acc = 0;

    // 遍历训练集
    for( auto& batch : loader)
    {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).view({-1});

        auto output = network.forward(data);
        auto loss = torch::nll_loss(output, targets);
        assert(!std::isnan(loss.template item<float>()));

        auto acc = output.argmax(1).eq(targets).sum();
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();

        if(index++ % options.log_interval == 0){
            auto end = std::min(data_size, (index + 1)*options.train_batch_size);
            std::cout << "Train Epoch:" << epoch << " " << end << "/" << data_size 
                                << "\tLoss:" << Loss/end << "\tAcc:" << Acc/end << std::endl;
        }
    }
}

template<typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size)
{
    size_t index = 0;
    network.eval();
    torch::NoGradGuard no_grad;
    float Loss=0, Acc=0;

    for(const auto& batch: loader)
    {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device);

        auto output = network.forward(data);
        auto loss = torch::nll_loss(output, targets.squeeze());
        assert(!std::isnan(loss.template item<float>()));

        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    if(index++ % options.log_interval ==0){
        std::cout << "Test Loss:" << Loss / data_size  << "\tAcc:" << Acc/ data_size << std::endl;
    }
        
}

int main(void)
{
    torch::manual_seed(0);

    auto data = readInfo();

    auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_set), options.train_batch_size);

    auto test_set = CustomDataset(data.second).map(torch::data::transforms::Stack<>());
    auto test_size = test_set.size().value();
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_set), options.test_batch_size);

    // std:: cout << train_size << std::endl;
    // std:: cout << test_size << std::endl;


    Network network;
    network.to(options.device);

    // torch::Tensor img = torch::randn({10, 1, 28, 28}, torch::requires_grad());
    // auto out = network.forward(img);
    // std::cout << out.sizes() << std::endl;
    // std::cout << out << std::endl;

    torch::optim::SGD optimizer(network.parameters(), torch::optim::SGDOptions(0.001).momentum(0.1));

    for(size_t i=0; i< options.iterations; ++i)
    {
        train(network, *train_loader, optimizer, i+1, train_size);
        std::cout << std::endl;
        test(network, *test_loader, test_size);
        std::cout << std::endl;
    }

    std::cout << "test" << std::endl;
    return 0;
}