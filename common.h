#pragma once

#include <stdio.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define checkCUDNN(status)                                                                         \
    {                                                                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                                      \
            printf("file : %s, func : %s, line: %d,CUDNN failure\nError:%s\n", __FILE__, __func__, \
                   __LINE__, cudnnGetErrorString(status));                                         \
        }                                                                                          \
    }

#define checkCUDA(status)                                                                         \
    {                                                                                             \
        if (status != cudaSuccess) {                                                              \
            printf("file : %s, func : %s, line: %d,CUDA failure\nError:%s\n", __FILE__, __func__, \
                   __LINE__, cudaGetErrorString(status));                                         \
        }                                                                                         \
    }

#define checkCUDAErrors(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define checkCUBLASErrors(stat) \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

typedef union {
    float f;
    int32_t i;
} FP32INT32_U;

typedef union {
    double f;
    int64_t i;
} FP64INT64_U;

// rand init
static float rand_in_range(float min, float max) {
    return ((max - min) * (rand() / (float)RAND_MAX) + min);
}

template <typename T>
void rand_data(T *data, int num, float min, float max) {
    for (int i = 0; i < num; i++) {
        float temp = rand_in_range(min, max);
        *(data + i) = (T)temp;
    }
}
// rand init

int getSizeInBytes(int num, cudnnDataType_t dataType) {
    switch (dataType) {
        case (CUDNN_DATA_FLOAT): return num * sizeof(float);
        case (CUDNN_DATA_HALF): return num * sizeof(half);
        case (CUDNN_DATA_INT8): return num * sizeof(int8_t);
        case (CUDNN_DATA_INT32): return num * sizeof(int32_t);
        default:
            printf("dataType not support\n");
            return -1;
            break;
    }

    return 0;
}

// malloc Func
int cudnnMallocInBytes(void **p, int size) {
    cudaMalloc(p, size);
    cudaMemset(*p, 0, size);
    return 0;
}

int cudnnMallocHostInBytes(void **p, int size) {
    cudaMallocHost(p, size);
    cudaMemset(*p, 0xff, size);  // default set to -Nan
    return 0;
}
// malloc Func

int cudnnMallocHost(void **p, int num, cudnnDataType_t dataType) {
    int size = getSizeInBytes(num, dataType);
    cudnnMallocHostInBytes(p, size);
    return 0;
}

// rand func
float myRand(void) {
    // use time as seed
    static int useOnce = 1;
    if (useOnce == 1) {
        useOnce += 1;
        srand(1658142400);
    }
    return rand();
}

float randFloatInRange(float minrange, float maxrange) {
    return (float)((maxrange - minrange) * (myRand() / (float)RAND_MAX) + minrange);
}

int randIntInRange(int minrange, int maxrange) {
    return (int)((maxrange - minrange) * (myRand() / (float)RAND_MAX) + minrange);
}

/* compare */
template <typename T1>
static typename std::enable_if<std::is_integral<T1>::value, bool>::type compare(T1 *data_cpu,
                                                                                T1 *data_gpu,
                                                                                size_t len) {
    float err = 0;
    float error_gpu_tensor = 0.0;
    float err_max_gpu_tensor = 0.0;
    float err_min_gpu_tensor = 0.0;
    for (size_t i = 0; i < len; i++) {
        float device_data = data_gpu[i];
        float baseline = data_cpu[i];
        if (fabs(device_data - baseline) > 1e-5) {
            printf("index: %ld, CPU data: %9f, GPU data: %9f, GPU err:%9f\n", i, baseline,
                   device_data, fabs(device_data - baseline));
        }
        err = fabs(device_data - baseline);
        if (i == 0) {
            err_max_gpu_tensor = err_min_gpu_tensor = err;
        } else {
            err_max_gpu_tensor = err_max_gpu_tensor < err ? err : err_max_gpu_tensor;
            err_min_gpu_tensor = err_min_gpu_tensor > err ? err : err_min_gpu_tensor;
        }
        error_gpu_tensor += err;
    }
    printf("<=GPU=> max err: %9f, min err: %9f, average err: %9f\n", err_max_gpu_tensor,
           err_min_gpu_tensor, error_gpu_tensor / len);
    return true;
}

template <typename T1>
static typename std::enable_if<std::is_floating_point<T1>::value, bool>::type compare(T1 *data_cpu,
                                                                                      T1 *data_gpu,
                                                                                      size_t len) {
    float err = 0;
    float error_gpu_tensor = 0.0;
    float err_max_gpu_tensor = 0.0;
    float err_min_gpu_tensor = 0.0;
    for (size_t i = 0; i < len; i++) {
        float device_data = data_gpu[i];
        float baseline = data_cpu[i];
        if (fabs(device_data - baseline) > 1e-5) {
            printf("index: %ld, CPU data: %9f, GPU data: %9f, GPU err:%9f\n", i, baseline,
                   device_data, fabs(device_data - baseline));
        }
        err = fabs(device_data - baseline);
        if (i == 0) {
            err_max_gpu_tensor = err_min_gpu_tensor = err;
        } else {
            err_max_gpu_tensor = err_max_gpu_tensor < err ? err : err_max_gpu_tensor;
            err_min_gpu_tensor = err_min_gpu_tensor > err ? err : err_min_gpu_tensor;
        }
        error_gpu_tensor += err;
    }
    printf("<=GPU=> max err: %9f, min err: %9f, average err: %9f\n", err_max_gpu_tensor,
           err_min_gpu_tensor, error_gpu_tensor / len);
    return true;
}

template <typename T1>
static typename std::enable_if<std::is_same<nv_half, T1>::value, bool>::type compare(T1 *data_cpu,
                                                                                     T1 *data_gpu,
                                                                                     size_t len) {
    float err = 0;
    float error_gpu_tensor = 0.0;
    float err_max_gpu_tensor = 0.0;
    float err_min_gpu_tensor = 0.0;
    for (size_t i = 0; i < len; i++) {
        float baseline = __half2float(data_cpu[i]);
        float device_data = __half2float(data_gpu[i]);
        if (fabs(device_data - baseline) > 1e-5) {
            printf("index: %ld, CPU data: %9f, GPU data: %9f, GPU err:%9f\n", i, baseline,
                   device_data, fabs(device_data - baseline));
        }
        err = fabs(device_data - baseline);
        if (i == 0) {
            err_max_gpu_tensor = err_min_gpu_tensor = err;
        } else {
            err_max_gpu_tensor = err_max_gpu_tensor < err ? err : err_max_gpu_tensor;
            err_min_gpu_tensor = err_min_gpu_tensor > err ? err : err_min_gpu_tensor;
        }
        error_gpu_tensor += err;
    }
    printf("<=GPU=> max err: %9f, min err: %9f, average err: %9f\n", err_max_gpu_tensor,
           err_min_gpu_tensor, error_gpu_tensor / len);
    return true;
}

template <typename DTYPE>
bool diff3_compare(const DTYPE *cpu_data, const DTYPE *gpu_data, const DTYPE *teco_data, int M,
                   int N) {
    bool passed = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            DTYPE cpu_value = cpu_data[i * N + j];
            DTYPE gpu_value = gpu_data[i * N + j];
            DTYPE teco_value = teco_data[i * N + j];
            double gpu_err = fabs((double)cpu_value - (double)gpu_value);
            double teco_err = fabs((double)cpu_value - (double)teco_value);
            if (gpu_err < teco_err) {
                printf("cpu_data[%d][%d]: %f, gpu_data[%d][%d]: %f, teco_data[%d][%d]: %f\n", i, j,
                       cpu_value, i, j, gpu_value, i, j, teco_value);
                printf("gpu_err: %f, teco_err: %f\n", gpu_err, teco_err);
                passed = false;
                break;
            }
        }
    }
    return passed;
}

#define MAX_ULP_ERR (0)
template <typename DTYPE>
bool bitwise_compare(const DTYPE *gpu_data, const DTYPE *teco_data, int M, int N) {
    bool passed = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t gpu_value = 0;
            int64_t teco_value = 0;
            if (std::is_same<DTYPE, float>::value) {
                FP32INT32_U gpu_u;
                gpu_u.f = gpu_data[i * N + j];
                gpu_value = gpu_u.i;
                FP32INT32_U teco_u;
                teco_u.f = teco_data[i * N + j];
                teco_value = teco_u.i;
            } else if (std::is_same<DTYPE, double>::value) {
                FP64INT64_U gpu_u;
                gpu_u.f = gpu_data[i * N + j];
                gpu_value = gpu_u.i;
                FP64INT64_U teco_u;
                teco_u.f = teco_data[i * N + j];
                teco_value = teco_u.i;
            } else {
                gpu_value = (int64_t)gpu_data[i * N + j];
                teco_value = (int64_t)teco_data[i * N + j];
            }

            unsigned long int ulp_err = abs(gpu_value - teco_value);
            if (ulp_err > MAX_ULP_ERR) {
                printf("gpu_data[%d][%d]: %lld, teco_data[%d][%d]: %lld\n", i, j, gpu_value, i, j,
                       teco_value);
                printf("ulp_err: %lld\n", ulp_err);
                passed = false;
                break;
            }
        }
        if (passed == false) {
            break;
        }
    }
    return passed;
}

/* run CPU API */
template <typename AT, typename BT, typename DT>
static void MatmulCPU_Impl(AT *matA, BT *matB, DT *matC, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            long double temp = 0;
            for (int w = 0; w < k; w++) {
                double d0 = double(matA[i * k + w]);
                double d1 = double(matB[j * k + w]);
                temp += (long double)d0 * (long double)d1;
            }
            double tmp_val = temp;
            matC[i * n + j] = (DT)tmp_val;
        }
    }
}