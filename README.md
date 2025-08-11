--build
    nvcc -arch=sm_100a fp4_demo_largeK.cu -o demo
--run
    ./demo > test.log
--compare
    python3 compare_gpu.py test.log

CPU计算为float模拟，与GPU fp4计算存在一定误差。