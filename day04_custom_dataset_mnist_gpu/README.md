pytorch c++ LeNet 训练MNIST数据集, cuda-10

1,执行 python mnist_download.py.保存训练数据,<图片和标签> <br/>
&nbsp;&nbsp;&nbsp;&nbsp;训练图片保存到./data/train/images/*.jpg <br/>
&nbsp;&nbsp;&nbsp;&nbsp;txt文件保存到./data/train/label.txt <br/>
  注意,如果采用Ubuntu系统,txt文件会多出一个换行符

2, 执行make,编译pytorch c++网络模型
  注意,程序设置为cuda上,注意makefile文件

3, 执行/bin/demo, 在pytorch c++ 的环境中执行LeNet网络训练网络,结果如下

```
Train Epoch:1 128/45000		Loss:0.123541	Acc:0.03429  
Train Epoch:1 6528/45000	Loss:0.0390586	Acc:0.158548
Train Epoch:1 12928/45000	Loss:0.021473	Acc:0.272394
```
