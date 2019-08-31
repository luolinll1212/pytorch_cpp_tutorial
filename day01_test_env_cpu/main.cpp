#include <torch/script.h>
#include <ATen/ATen.h>

#include <iostream>

using namespace std;
using namespace at;

int main(int argc, const char* argv[])
{
    at::Tensor a = at::ones({2,2}, at::kInt);
    std::cout << a << std::endl;

    std::cout << "ok" << std::endl;
    return 0;
}