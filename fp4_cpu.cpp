#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>

#define M 128
#define N 32
#define K 64
// ========================
static const float fp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  // 0x0 ~ 0x3
     2.0f,  3.0f,  4.0f,  6.0f,  // 0x4 ~ 0x7
    -0.0f, -0.5f, -1.0f, -1.5f,  // 0x8 ~ 0xB
    -2.0f, -3.0f, -4.0f, -6.0f   // 0xC ~ 0xF
};

float fp4_to_float(uint8_t bits) {
    return fp4_lut[bits & 0x0F];
}

// 从打包的 uint8_t 中提取两个 fp4 并转为 float
void unpack_fp4(uint8_t packed, float &val0, float &val1) {
    val0 = fp4_to_float((packed >> 4) & 0x0F);  // 高4位
    val1 = fp4_to_float(packed & 0x0F);         // 低4位
}

// GEMM: C = A @ B, 其中 A(M,K), B(K,N), C(M,N)
// M=32, K=256, N=32
// ========================
void gemm_fp4_to_float(
    const std::vector<uint8_t>& A_packed,  // size = M*K/2
    const std::vector<uint8_t>& B_packed,  // size = N*K/2
    std::vector<float>& C                  // size = M*N
) {
    C.assign(M * N, 0.0f);

    // 临时 buffer 存储解码后的 A 和 B（可选：也可在循环中实时解码）
    std::vector<float> A_float(M * K);
    std::vector<float> B_float(K * N);
    std::vector<float> B_T(K * N);

    // 解码 A: M*K, 每个 uint8_t 包含两个 fp4
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; k += 2) {
            uint8_t packed = A_packed[i * (K/2) + k/2];
            float val0, val1;
            unpack_fp4(packed, val0, val1);
            A_float[i * K + k]     = val0;
            A_float[i * K + k + 1] = val1;
        }
    }

    // 解码 B: N*K
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            uint8_t packed = B_packed[n * (K/2) + k/2];
            float val0, val1;
            unpack_fp4(packed, val0, val1);
            B_float[n * K + k]     = val0;
            B_float[n * K + k + 1] = val1;
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B_T[k * N + n] = B_float[n * K + k];
        }
    }

    // 执行 GEMM: C[i][j] += A[i][k] * B[k][j]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_float[i * K + k] * B_T[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ========================
// 主函数：测试
// ========================
int main() {

    // 分配 packed 数据：每两个 fp4 存在一个 uint8_t 中
    std::vector<uint8_t> A_packed(M * K / 2);
    std::vector<uint8_t> B_packed(N * K / 2);

    // 初始化 A 和 B 为一些有意义的 fp4 值（例如 0~15）
    // 这里用简单模式：A[i][k] = (i + k) % 16, B[k][j] = (k - j + 16) % 16
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; k += 2) {
            uint8_t val0 = (i + k) % 16;
            uint8_t val1 = (i + k + 1) % 16;
            // uint8_t val0 = 0x33;
            // uint8_t val1 = 0x33;
            A_packed[i * (K/2) + k/2] = (val0 << 4) | (val1 & 0x0F);
        }
    }
    printf("host_A\n");
    for(int m = 0; m < M; m++) {
        printf("%3d : ",m);
        for(int n = 0; n < K/2; n++) {
          printf("%2x ",A_packed[m * K/2 + n]);
        }
    printf("\n");
    }
    printf("\n");
 
    printf("host_B\n");
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; k += 2) {
        uint8_t val0 = (j - k + 16) % 15;
        uint8_t val1 = (j - k - 1 + 16) % 15;
        B_packed[j * (K/2) + k/2] = (val0 << 4) | (val1 & 0x0F);
      }
    }
    for (int k = 0; k < N; ++k) {
        printf("%3d : ",k);
        for (int j = 0; j < K; j += 2) {
          printf("%2x ",B_packed[k * (K/2) + j/2]);
        }
    printf("\n");
    }
    printf("\n");
 
    // 输出矩阵 C
    std::vector<float> C(M * N);

    // 执行计算
    gemm_fp4_to_float(A_packed, B_packed, C);

    // 打印结果（只打印左上角 5x5）
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << std::setw(9) << C[i * N + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    for(int m = 0; m < M; m++) {
        printf("%3d : ",m);
        for(int n = 0; n < N; n++) {
          printf("%08.2f ",C[m * N + n]);
        }
      printf("\n");
    }



    return 0;
}