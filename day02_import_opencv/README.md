本节只实现调用opencv,加载图片

调用opencv有两种方式

方法一,通过apt-get方式安装
        ```sudo apt-get install libopencv-dev python-opencv ``` <br/>
CMakeLists.txt文件编写为 find_package(OpenCV REQUIRED)

方法二,通过编译源码方式调用,具体过程查看[地址](https://blog.csdn.net/luolinll1212/article/details/88376414)