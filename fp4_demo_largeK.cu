#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>
#include "common.h"
#include "fp4_cpu.h"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>
// TensorCore05 使用 MXF4NVF4 (4-bit Float) 格式示例
#define M 128
#define N 32
#define K 256
#define BLOCKSCALE_NX 2   //scale_vec::2X //指令内部分手动设置

union SmemDescriptor
{
  uint64_t desc_ = 0;
  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    
    uint64_t start_address_         : 14,    // 14  [0,14),     // start_address, bit [0,14), 4LSB not included
                                    : 2,     // 2   [14,16)  
             leading_byte_offset_   : 14,    // 14  [16,30)     // leading dimension byte offset, bit [16,30), 4LSB not included
                                    : 2,     // 2   [30,32)  
             stride_byte_offset_    : 14,    // 14  [32,46)     // stride dimension byte offset, bit [32,46), 4LSB not included
             fixed_001_             : 3,     // 3   [46,49)
             base_offset_           : 3,     // 3   [49,52)     // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
             fixed_b0_              : 1,     // 1   [52,53) 
             fixed_b00000000_       : 8,     // 8   [53,61)
             layout_type_           : 3;     // 3   [61,64)     // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
  };

  // Decay to a uint64_t
   __forceinline__ __host__ __device__ constexpr
  operator uint64_t() const noexcept { return desc_; }
};

union InstrDescriptorBlockScaled
{
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
             sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
                            : 1,  //
             b_sf_id_       : 2,  // bit [ 4, 6) : Matrix B Scale Factor ID
                            : 1,  //
             a_format_      : 3,  // bit [ 7, 9) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
             b_format_      : 2,  // bit [10,12) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
                            : 1,
             a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
             a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
             n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
             scale_format_  : 1,  // bit [23,24) : 0=E4M3, 1=E8M0
                            : 3,  //
             m_dim_         : 2,  // bit [27,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
             a_sf_id_       : 2,  // bit [29,31) : Matrix A Scale Factor ID
             k_dim_         : 1;  //
  };
 
  // Decay to a uint32_t
  __forceinline__ __host__ __device__ constexpr
  operator uint32_t() const noexcept { return desc_; }
};


__global__ void mma_on_tmem(uint8_t *mat_a, uint8_t *mat_b, uint8_t *mat_sfa, uint8_t *mat_sfb, float *mat_c) {
  int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t  mat_a_share[M * 32];
  __shared__ uint8_t  mat_b_share[N * 32];
  __shared__ uint8_t  sfa_share[M*16];
  __shared__ uint8_t  sfb_share[N*16];
  for (int i = 0; i < M * 16; ++i) sfa_share[i] = mat_sfa[i];
  for (int i = 0; i < N * 16; ++i) sfb_share[i] = mat_sfb[i];

  __syncthreads();

  __shared__ uint32_t s_tmem_ptr[1];
  __shared__ uint32_t s_tmem_scaleA_ptr[1];
  __shared__ uint32_t s_tmem_scaleB_ptr[1];
  unsigned tmem_addr   = (unsigned)__cvta_generic_to_shared(&s_tmem_ptr[0]);
  unsigned scaleA_tmem_addr = (unsigned)__cvta_generic_to_shared(&s_tmem_scaleA_ptr[0]);
  unsigned scaleB_tmem_addr = (unsigned)__cvta_generic_to_shared(&s_tmem_scaleB_ptr[0]);

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(tmem_addr),   "r"(32));

    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(scaleA_tmem_addr), "r"(32));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(scaleB_tmem_addr), "r"(32));
  }
  __syncthreads();

  for(int k_loop = 0; k_loop < K; k_loop += 64) {
    for (int i = 0; i < M; i++){
      for (int j = 0; j < 32; j++){
        if ((i/4)%2==0){
          mat_a_share[i*32 +j] = mat_a[i*K/2 +j + k_loop / 2];
        }else{
          int j_ = (j+16)%32;
          mat_a_share[i*32 +j] = mat_a[i*K/2 +j_ + k_loop / 2];
        }
      }
    }

    for (int i = 0; i < N; i++){
      for (int j = 0; j < 32; j++){
        if ((i/4)%2==0){
          mat_b_share[i*32 +j] = mat_b[i*K/2 +j + k_loop / 2];
        }else{
          int j_ = (j+16)%32;
          mat_b_share[i*32 +j] = mat_b[i*K/2 +j_ + k_loop / 2];
        }
      }
    }

    //构建 shared memory descriptor
    SmemDescriptor mat_a_desc{};
    mat_a_desc.start_address_ = ((unsigned)__cvta_generic_to_shared(&mat_a_share[0])) >> 4;
    mat_a_desc.leading_byte_offset_ = 16>>4;    //当前线程两个128bit之间地址差
    mat_a_desc.stride_byte_offset_ = 256>>4;    //0号和8号 首地址地址差

    mat_a_desc.fixed_001_      = 0xb001;
    mat_a_desc.base_offset_    = 0;
    mat_a_desc.fixed_b0_       = 0xb0;
    mat_a_desc.fixed_b00000000_= 0;
    mat_a_desc.layout_type_    = 6;             //


    // B 矩阵 构建 shared memory descriptor K-major 
    SmemDescriptor mat_b_desc{};
    mat_b_desc.start_address_ = ((unsigned)__cvta_generic_to_shared(&mat_b_share[0])) >> 4;
    mat_b_desc.leading_byte_offset_ = 16>>4;
    mat_b_desc.stride_byte_offset_ = 256>>4;

    mat_b_desc.fixed_001_      = 0xb001;
    mat_b_desc.base_offset_    = 0;
    mat_b_desc.fixed_b0_       = 0xb0;
    mat_b_desc.fixed_b00000000_= 0;
    mat_b_desc.layout_type_    = 6;

    SmemDescriptor mat_sfa_desc{};
    mat_sfa_desc.desc_=0;
    mat_sfa_desc.start_address_ = ((unsigned)__cvta_generic_to_shared(&sfa_share[0])) >> 4;
    mat_sfa_desc.leading_byte_offset_ = 0>>4;
    mat_sfa_desc.stride_byte_offset_ = 0>>4;
    mat_sfa_desc.fixed_001_      = 0xb001;
    mat_sfa_desc.base_offset_    = 0;
    mat_sfa_desc.fixed_b0_       = 0xb0;
    mat_sfa_desc.fixed_b00000000_= 0;
    mat_sfa_desc.layout_type_    = 0;

    SmemDescriptor mat_sfb_desc{};
    mat_sfb_desc.desc_=0;
    mat_sfb_desc.start_address_ = ((unsigned)__cvta_generic_to_shared(&sfb_share[0])) >> 4;
    mat_sfb_desc.leading_byte_offset_ = 0>>4;
    mat_sfb_desc.stride_byte_offset_ = 0>>4;
    mat_sfb_desc.fixed_001_      = 0xb001;
    mat_sfb_desc.base_offset_    = 0;
    mat_sfb_desc.fixed_b0_       = 0xb0;
    mat_sfb_desc.fixed_b00000000_= 0;
    mat_sfb_desc.layout_type_    = 0;


    //构建 Instr Descriptor
    InstrDescriptorBlockScaled desc = {};
    desc.desc_ = 0;
    desc.m_dim_ = 128>>7;
    desc.n_dim_ = 32>>3;
    desc.sparse_flag_=0;
    desc.sparse_id2_=0;
    desc.a_format_ = 1;
    desc.b_format_ = 1;
    desc.scale_format_ = 1;
    desc.a_major_ = 0;
    desc.b_major_ = 0;
    desc.a_negate_ = 0;
    desc.b_negate_ = 0;

    desc.a_sf_id_ = (s_tmem_scaleA_ptr[0] & 0xC0000000) >> 30;
    desc.b_sf_id_ = (s_tmem_scaleB_ptr[0] & 0xC0000000) >> 30;
    // 最终构造为 64bit 指令描述符（位于高32位）
    uint64_t idesc = (uint64_t(uint32_t(desc)) << 32);


    if(tid%32==0){  //single thread per warp 
        // printf(" Copy SMem to TMem   \n");

        asm volatile ("tcgen05.cp.cta_group::1.128x128b [%0], %1;"
        :
        : "r"(s_tmem_scaleA_ptr[0]),"l"(uint64_t(mat_sfa_desc)));

        asm volatile ("tcgen05.cp.cta_group::1.128x128b [%0], %1;"
        :
        : "r"(s_tmem_scaleB_ptr[0]),"l"(uint64_t(mat_sfb_desc)));
    }
    asm volatile ("tcgen05.fence::before_thread_sync;");
    __syncthreads();



    // 执行 MMA 指令
    uint32_t scaleC  = k_loop == 0 ? 0 : 1;
    // if (tid%128==0)printf(" run mma fp4   \n");

    // if (tid%32==0) {
    if (tid ==0) {

        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
            "}\n"
            :
            : "r"(s_tmem_ptr[0]), "l"(uint64_t(mat_a_desc)),
              "l"(uint64_t(mat_b_desc)), "r"(uint32_t(idesc>>32)),
              "r"(scaleC),
              "r"(s_tmem_scaleA_ptr[0]),
              "r"(s_tmem_scaleB_ptr[0])
        );
    }

    asm volatile("tcgen05.fence::before_thread_sync;");

    __syncthreads();
  }

  // 读取结果回寄存器
  uint32_t regD[32];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 { %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
               : "=r"(regD[0]), "=r"(regD[1]), "=r"(regD[2]), "=r"(regD[3]),
                 "=r"(regD[4]), "=r"(regD[5]), "=r"(regD[6]), "=r"(regD[7]),
                 "=r"(regD[8]), "=r"(regD[9]), "=r"(regD[10]), "=r"(regD[11]),
                 "=r"(regD[12]), "=r"(regD[13]), "=r"(regD[14]), "=r"(regD[15]),
                 "=r"(regD[16]), "=r"(regD[17]), "=r"(regD[18]), "=r"(regD[19]),
                 "=r"(regD[20]), "=r"(regD[21]), "=r"(regD[22]), "=r"(regD[23]),
                 "=r"(regD[24]), "=r"(regD[25]), "=r"(regD[26]), "=r"(regD[27]),
                 "=r"(regD[28]), "=r"(regD[29]), "=r"(regD[30]), "=r"(regD[31])
               : "r"(s_tmem_ptr[0]));
  for(int i = 0; i < 32; i++) {
    mat_c[32 * tid + i] = ((float*)regD)[i];
  }

  __syncthreads();
    unsigned int taddr_0, taddr_a, taddr_b;

    if (threadIdx.x < 32) {
      asm volatile("ld.shared.b32 %0, [%1];":"=&r"(taddr_0):"r"(tmem_addr));
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
          : : "r"(taddr_0), "r"(32));
      asm volatile("ld.shared.b32 %0, [%1];":"=&r"(taddr_a):"r"(scaleA_tmem_addr));
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
          : : "r"(taddr_a), "r"(32));
      asm volatile("ld.shared.b32 %0, [%1];":"=&r"(taddr_b):"r"(scaleB_tmem_addr));
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
          : : "r"(taddr_b), "r"(32));
    }
  __syncthreads();

}


float e2m1_to_float(uint8_t four_bits) {
  // 提取符号位
  bool sign = (four_bits >> 3) & 0x01;

  // 提取指数位
  uint8_t exp_bits = (four_bits >> 1) & 0x03;

  // 提取尾数位
  uint8_t mantissa_bit = four_bits & 0x01;

  float value;
  if (exp_bits == 0) {
      // Subnormal number: no implicit 1, exponent = 0
      value = mantissa_bit * 0.5f;
  } else {
      // Normal number: implicit 1
      float mantissa = 1.0f + mantissa_bit * 0.5f;
      int exponent = exp_bits - 1;  // bias = 1
      value = mantissa * powf(2.0f, exponent);
  }

  return sign ? -value : value;
}


// 从uint8_t中提取两个e2m1值并打印
void print_two_e2m1(uint8_t data) {
  // di4位是第一个e2m1值
  uint8_t first = data & 0x0F;
  // 低4位是第二个e2m1值
  uint8_t second = (data >> 4) & 0x0F;
  
  // 转换为十进制
  float val1 = e2m1_to_float(first);
  float val2 = e2m1_to_float(second);
  
  // 打印结果
  std::cout << std::fixed << std::setprecision(1) << val1 << " " << val2;
}

int main() {
    if(K % 64) {
      printf("Error: K is not a multiple of 64\n");
      return 1;
    }

    std::srand(std::time(nullptr));  // 设置当前时间为种子

    cudaLaunchConfig_t config;
    config.gridDim = 1;
    config.blockDim = 128;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    void *host_A = nullptr, *host_B = nullptr, *host_C = nullptr, *host_sfa = nullptr, *host_sfb = nullptr;
    cudnnMallocHostInBytes(&host_A, M * K * sizeof(uint8_t) / 2);
    cudnnMallocHostInBytes(&host_B, N * K * sizeof(uint8_t) / 2);
    cudnnMallocHostInBytes(&host_C, M * N * sizeof(float));
    cudnnMallocHostInBytes(&host_sfa, M * 16 * sizeof(uint8_t));
    cudnnMallocHostInBytes(&host_sfb, N * 16 * sizeof(uint8_t));

    uint8_t sca = 117+rand()%20;
    uint8_t scb = 117+rand()%20;
    // for(int i = 0; i < M * 16; i++) ((uint8_t *)host_sfa)[i] = sca;
    // for(int i = 0; i < N * 16; i++) ((uint8_t *)host_sfb)[i] = scb;
    for(int i = 0; i < M * 16; i++) ((uint8_t *)host_sfa)[i] = 127;
    for(int i = 0; i < N * 16; i++) ((uint8_t *)host_sfb)[i] = 127;

    // mat A B rand data
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; k += 2) {
        uint8_t val0 = rand() % 16;
        uint8_t val1 = rand() % 16;
        ((uint8_t *)host_A)[i * (K/2) + k/2] = (val0 << 4) | (val1 & 0x0F);
      }
    }

    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < K; k += 2) {
        uint8_t val0 = rand() % 16;
        uint8_t val1 = rand() % 16;
        ((uint8_t *)host_B)[i * (K/2) + k/2] = (val0 << 4) | (val1 & 0x0F);
      }
    }

    {
      std::cout <<"shape:"<<M<<" "<<N<<" "<<K<<std::endl;
      std::cout <<"kind::mxf4nvf4"<<std::endl;
      std::cout <<"Mat_A(K-major:128x64):"<<std::endl;
      for (int i = 0; i < M; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int k = 0; k < K; k += 2) {
          print_two_e2m1(((uint8_t *)host_A)[i * (K/2) + k/2]);
          std::cout <<" ";
        }
        std::cout <<"]"<<std::endl;
      }

        std::cout <<"Mat_A 0x:"<<std::endl;
      for (int i = 0; i < M; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int k = 0; k < K; k += 2) {
          printf("0x%02X , ", ((uint8_t *)host_A)[i * (K/2) + k/2]);
        }
        std::cout <<"]"<<std::endl;
      }

        std::cout <<"Mat_B(K-major:32x64):"<<std::endl;
      for (int i = 0; i < N; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int k = 0; k < K; k += 2) {
          print_two_e2m1(((uint8_t *)host_B)[i * (K/2) + k/2]);
          std::cout <<" ";
        }
        std::cout <<"]"<<std::endl;
      }

        std::cout <<"Mat_B 0x:"<<std::endl;
      for (int i = 0; i < N; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int k = 0; k < K; k += 2) {
          printf("0x%02X , ", ((uint8_t *)host_B)[i * (K/2) + k/2]);
        }
        std::cout <<"]"<<std::endl;
      }

        std::cout <<"scale A (scale_vec::"<<BLOCKSCALE_NX<<"X 、e8m0) 0x:"<<std::endl;
      for (int i = 0; i < M; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int j = 0; j < BLOCKSCALE_NX; j ++) {
          printf("0x%02X , ", ((uint8_t *)host_sfa)[i * 16 + j]);
        }
        std::cout <<"]"<<std::endl;
      }

        std::cout <<"scale B (scale_vec::"<<BLOCKSCALE_NX<<"X 、e8m0) 0x:"<<std::endl;
      for (int i = 0; i < N; ++i) {
        std::cout << std::setw(3) <<i<<":[ ";
        for (int j = 0; j < BLOCKSCALE_NX; j ++) {
          printf("0x%02X , ", ((uint8_t *)host_sfb)[i * 16 + j]);
        }
        std::cout <<"]"<<std::endl;
      }

    }
 
    uint8_t *dev_A, *dev_B, *dev_C, *dev_sfa, *dev_sfb;
    cudaMalloc(&dev_A, M * K * sizeof(uint8_t) / 2);
    cudaMalloc(&dev_B, N * K * sizeof(uint8_t) / 2);
    cudaMalloc(&dev_C, M * N * sizeof(float));
    cudaMalloc(&dev_sfa, M * 16 * sizeof(uint8_t));
    cudaMalloc(&dev_sfb, N * 16 * sizeof(uint8_t));
    checkCUDAErrors(cudaMemcpy(dev_A, host_A, M * K * sizeof(uint8_t) / 2, cudaMemcpyHostToDevice));
    checkCUDAErrors(cudaMemcpy(dev_B, host_B, N * K * sizeof(uint8_t) / 2, cudaMemcpyHostToDevice));
    checkCUDAErrors(cudaMemcpy(dev_sfa, host_sfa, M * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCUDAErrors(cudaMemcpy(dev_sfb, host_sfb, N * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    auto status = cudaLaunchKernelEx(&config, mma_on_tmem,
                                     (uint8_t *)dev_A,
                                     (uint8_t *)dev_B,
                                     (uint8_t *)dev_sfa,
                                     (uint8_t *)dev_sfb,
                                     (float *)dev_C);
    cudaDeviceSynchronize();

    checkCUDAErrors(cudaMemcpy(host_C, dev_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    std::cout <<"Mat_C:"<<std::endl;
    for(int m = 0; m < M; m++) {
      printf(" %3d : ",m);
      for(int n = 0; n < N; n++) {
        printf("%08.2f ",((float *)host_C)[m * N + n]);
      }
      printf("\n");
    }

    // 输出矩阵 C
    std::vector<float> C(M * N);
    C.assign(M * N, 0.0f);

    // 执行计算
    gemm_fp4_to_float<M, N, K>((uint8_t *)host_A, (uint8_t *)host_B, C);

    bool cmp_fail = false;
    std::cout <<"Mat_C_cpu:"<<std::endl;
    for(int m = 0; m < M; m++) {
      printf(" %3d : ",m);
      for(int n = 0; n < N; n++) {
        printf("%08.2f ",C[m * N + n]);
        if (((float *)host_C)[m * N + n] != C[m * N + n]) {
          cmp_fail = true;
        }
      }
      printf("\n");
    }

    std::cerr << "Compare " << (cmp_fail ? "fail !" : "success !") << std::endl;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFree(dev_sfa);
    cudaFree(dev_sfb);
    cudaFreeHost(host_A);
    cudaFreeHost(host_B);
    cudaFreeHost(host_C);
    cudaFreeHost(host_sfa);
    cudaFreeHost(host_sfb);

    return 0;
}
