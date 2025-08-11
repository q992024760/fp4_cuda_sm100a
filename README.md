--build
    nvcc -arch=sm_100a fp4_demo_largeK.cu -o demo
--run
    ./demo > test.log
--compare
    python3 compare_gpu.py test.log

--K扩展
    修改程序中K 值，应为64的倍数
--Scale
    SCALE_FORMAT 0  // E4M3
    SCALE_FORMAT 1  // E8M0

CPU计算为float模拟，与GPU fp4计算存在一定误差。