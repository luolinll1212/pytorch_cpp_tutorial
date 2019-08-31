#-*- conding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import traceback

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

print("-"*50)

def check_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    fp = open(file_path,"w")
    fp.close()

def check_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            print("delete --",os.path.join(path, filename))
            os.remove(os.path.join(path, filename))
        os.removedirs(path)
    os.makedirs(path)


# 1,查看训练数据的大小
print("训练数据图片的形状{}".format(mnist.train.images.shape))
print("训练数据图片标签的形状{}".format(mnist.train.labels.shape))
print("-"*50)

# 2,查看测试数据的大小
print("测试数据图片的形状{}".format(mnist.test.images.shape))
print("测试数据图片标签的形状{}".format(mnist.test.labels.shape))
print("-"*50)


# 3,创建保存文件夹
path = "./data"
if os.path.exists(path):
    pass
else:
    os.makedirs(path)

# 4,设置保存路径
images_path = os.path.join(path, "train/images")   # 保存的图片
file_path = os.path.join(path, "train/label.txt")  # 保存的标签
print(images_path)
print(file_path)

# 5,检查文件和目录是否存在
check_dir(images_path)    # 检查保存的文件夹是否存在
check_file(file_path)     # 检查保存的标签是否存在

# exit()

# 6, 保存图片和标签
num = 55000
with open(file_path, "w+") as f:
    for i in range(num):
        # 拿到图片
        im_data = np.array(np.reshape(mnist.train.images[i], (28, 28)) * 255, dtype=np.int8)
        img = Image.fromarray(im_data, "L")

        # 拿到标签
        label = mnist.train.labels[i]

        # 保存
        img.save(os.path.join(images_path, "{0}.jpg".format(i)))

        train_type = "train"
        test_type = "test"
        if i<45000:
            line = "{0}/{1}.jpg {2} {3}\n".format(images_path,i, label, train_type)
        else:
            line = "{0}/{1}.jpg {2} {3}\n".format(images_path,i, label, test_type)
        f.write(line)
