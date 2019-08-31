.PHONY:all clean

# bin为cmake输出文件
PATH_BIN=bin
#pytorch_lib为编译文件,官网下载,然后解压
pytorch_lib=/home/rose/software/libtorch_cpu
#pytorch_lib_cuda=/home/rose/software/libtorch_gpu_cuda_100

all:
	@cd ./$(PATH_BIN) && cmake -DCMAKE_PREFIX_PATH=$(pytorch_lib) .. && make

clean:
	rm -rf $(PATH_BIN)
	mkdir $(PATH_BIN)

