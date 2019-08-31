pytorch c++ LeNet 训练MNIST数据集

1,执行 python mnist_download.py.保存训练数据,<图片和标签> <br/>
&nbsp;&nbsp;&nbsp;&nbsp;训练图片保存到./data/train/images/*.jpg <br/>
&nbsp;&nbsp;&nbsp;&nbsp;txt文件保存到./data/train/label.txt <br/>
注意,如果采用Ubuntu系统,txt文件会多出一个换行符

2, 执行make,编译pytorch c++网络模型

3, 执行/bin/demo, 在pytorch c++ 的环境中执行LeNet网络训练网络,结果如下

```
Train Epoch:1 128/45000	Loss:0.144541	Acc:0.03125
Train Epoch:1 6528/45000	Loss:0.0398586	Acc:0.158548
Train Epoch:1 12928/45000	Loss:0.0365473	Acc:0.194384
Train Epoch:1 19328/45000	Loss:0.0346163	Acc:0.230495
Train Epoch:1 25728/45000	Loss:0.0330105	Acc:0.265314
Train Epoch:1 32128/45000	Loss:0.0316326	Acc:0.294976
Train Epoch:1 38528/45000	Loss:0.0303662	Acc:0.322363
Train Epoch:1 44928/45000	Loss:0.0292117	Acc:0.35045

Test Loss:0.0128972	Acc:7.1733

Train Epoch:2 128/45000	Loss:0.0111592	Acc:0.273438
Train Epoch:2 6528/45000	Loss:0.0201234	Acc:0.550858
Train Epoch:2 12928/45000	Loss:0.0194903	Acc:0.570777
Train Epoch:2 19328/45000	Loss:0.0190157	Acc:0.583454

```
