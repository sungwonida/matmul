//≥™¿Ã∫Í ƒ⁄µÂ øπΩ√

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>

#define HAVE_AVX512 1  // 0: FALSE, 1: TRUE


class Timer {
    struct timespec s_;
public:
    Timer() { tic(); }
    void tic() {
        clock_gettime(CLOCK_REALTIME, &s_);
    }

    double toc() {
        struct timespec e;
        clock_gettime(CLOCK_REALTIME, &e);
        return (double)(e.tv_sec - s_.tv_sec) + 1e-9 * (double)(e.tv_nsec - s_.tv_nsec);
    }
};

namespace david {
namespace algorithms {
namespace math {

void matmult_cuda(int r, int c, int k, const float* h_A, const float* h_B, float* h_C, int impl = 0);

template <typename T>
union VecPtr {
    const float* data;
    T* p;
};

#if HAVE_AVX512
static inline __m512 reduce_ma_pack16(int k, int n, const float* A, const float* B) {
    __m512 buf;
    memcpy(&buf, B, sizeof(__m512));
    __m512 x = _mm512_mul_ps(_mm512_set1_ps(A[0]), buf);
    int offset = n;

    for (int i = 1; i < k; ++i) {
        memcpy(&buf, B + offset, sizeof(__m512));
        // Fused-multiply-add is more faster and accurate
        // Because it performs the rounding once instead of twice
        // x = _mm512_add_ps(x, _mm512_mul_ps(_mm512_set1_ps(a[i]), buf));
        x = _mm512_fmadd_ps(_mm512_set1_ps(A[i]), buf, x);
        offset += n;
    }
}
#endif

static inline __m256 reduce_ma_pack8(int k, int n, const float* A, const float* B) {
    __m256 buf;
    memcpy(&buf, B, sizeof(__m256));
    __m256 x = _mm256_mul_ps(_mm256_broadcast_ss(&A[0]), buf);
    int offset = n;

    for (int i = 1; i < k; ++i) {
        memcpy(&buf, B + offset, sizeof(__m256));
        x = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[i]), buf, x);
        offset += n;
    }

    return x;
}

static inline __m128 reduce_ma_pack4(int k, int n, const float* A, const float* B) {
    __m128 buf;
    memcpy(&buf, B, sizeof(__m128));
    __m128 x = _mm_mul_ps(_mm_broadcast_ss(&A[0]), buf);
    int offset = n;

    for (int i = 1; i < k; ++i) {
        memcpy(&buf, B + offset, sizeof(__m128));
        x = _mm_fmadd_ps(_mm_broadcast_ss(&A[i]), buf, x);
        offset += n;
    }

    return x;
}

static inline float work_hard(int k, int n, const float* A, const float* B) {
    float x = .0;
    int irow = 0;

    for (int i = 0; i < k; ++i) {
        x += A[i] * B[irow];
        irow += n;
    }

    return x;
}

static inline void compute_row(int k, int n, int n_quant16, int n_quant8, int n_quant4, int n_remainder, const float* A_row, const float* B, float* C_row) {
    int stride = 0;

    /**
     * 1) Compute C_row = A_row * B
     * 2) Copy back the result
     * 3) Update the index
     */
#if HAVE_AVX512
    for (int i = 0; i < n_quant16; ++i) {
        __m512 result = reduce_ma_pack16(k, n, A_row, B + stride);
        VecPtr<__m512> tmp;
        tmp.p = &result;
        memcpy(C_row + stride, tmp.data, 16 * sizeof(float));
        stride += 16;
    }
#endif

    for (int i = 0; i < n_quant8; ++i) {
        __m256 result = reduce_ma_pack8(k, n, A_row, B + stride);
        VecPtr<__m256> tmp;
        tmp.p = &result;
        memcpy(C_row + stride, tmp.data, 8 * sizeof(float));
        stride += 8;
    }

    for (int i = 0; i < n_quant4; ++i) {
        __m128 result = reduce_ma_pack4(k, n, A_row, B + stride);
        VecPtr<__m128> tmp;
        tmp.p = &result;
        memcpy(C_row + stride, tmp.data, 4 * sizeof(float));
        stride += 4;
    }

    for (int i = 0; i < n_remainder; ++i) {
        float result = work_hard(k, n, A_row, B + stride);
        C_row[stride] = result;
        stride += 1;
    }
}

void matmult_simd(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c) {

    // column
#if HAVE_AVX512
    int n_quant16 = n / 16;
#else
    int n_quant16 = 0;
#endif
    int n_quant8 = (n - n_quant16 * 16) / 8;
    int n_quant4 = (n - n_quant16 * 16 - n_quant8 * 8) / 4;
    int n_remainder = n - n_quant16 * 16 - n_quant8 * 8 - n_quant4 * 4;

    // row
#if HAVE_AVX512
    int m_quant16 = m / 16;
#else
    int m_quant16 = 0; // This doesn't really have to be
#endif
    int m_quant8 = (m - m_quant16 * 16) / 8;
    int m_quant4 = (m - m_quant16 * 16 - m_quant8 * 8) / 4;
    int m_remainder = m - m_quant16 * 16 - m_quant8 * 8 - m_quant4 * 4;

    // std::cout << "(" << m << " x " << k << "), (" << k << " x " << n << ")" << std::endl;
    // std::cout << "n_quant16: " << n_quant16 << std::endl;
    // std::cout << "n_quant8: " << n_quant8 << std::endl;
    // std::cout << "n_quant4: " << n_quant4 << std::endl;
    // std::cout << "n_remainder: " << n_remainder << std::endl;
    // std::cout << std::endl;

    int sa = 0, sc = 0; // strides

    for (int i = 0; i < m_quant16; ++i) {
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }

    for (int i = 0; i < m_quant8; ++i) {
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }

    for (int i = 0; i < m_quant4; ++i) {
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }

    for (int i = 0; i < m_remainder; ++i) {
        compute_row(k, n, n_quant16, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }
}
} // math
} // algorithms
} // david

void matmult_opencv(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c)
{
    cv::Mat A = cv::Mat(m, k, CV_32F, const_cast<float *>(mat_a)).clone();
    cv::Mat B = cv::Mat(k, n, CV_32F, const_cast<float *>(mat_b)).clone();
    cv::Mat C = A * B;
    memcpy(mat_c, C.data, m * n * sizeof(float));
}

void matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c)
{
    /*
        == input ==
        mat_a: m x k matrix
        mat_b: k x n matrix

        == output ==
        mat_c: m x n matrix (output)
    */

    for (int i1=0; i1<m; i1++) {
        for (int i2=0; i2<n; i2++) {
            mat_c [n*i1 + i2] = 0;
            for (int i3=0; i3<k; i3++) {
                mat_c[n*i1 + i2] += mat_a[i1 * k + i3] * mat_b[i3 * n + i2];
            }
        }
    }
}

void genmat(int n, int m, std::vector<float>& mat)
{
    srand(time(0));
    // srand(0); //DONE: see if it reproduce the result of reference (naive one)
    mat.resize(n * m);
    for (int i=0; i < mat.size(); i++) mat[i] = rand() % 100;
}

bool validate(std::vector<float> &mat_a, std::vector<float> &mat_b, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat_a[i * n + j] != mat_b[i * n + j]) {
                printf("(%d, %d): %f, %f\n", i, j, mat_a[i * n + j], mat_b[i * n + j]);
                return false;
            }
        }
    }

    return true;
}

void dumpmat(int n, int m, std::vector<float>& mat)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
            printf("%f ", mat[i * m + j]);
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;

    int m = 128;
    int k = 128;
    int n = 128;

    genmat(m, k, A);
    genmat(k, n, B);
    genmat(m, n, C);

    // Create a reference for sanity check
    std::vector<float> R;
    genmat(m, n, R);

    Timer t;
    double elapsed;
    const int iteration = 1000;

    /*********************
     * Naive version (CPU)
     */
    elapsed = 0.;
    for (int i = 0; i < iteration; i++) {
        t.tic();
        matmult(m, n, k, &A[0], &B[0], &R[0]);
        elapsed += t.toc();
    }
    printf("NAIVE %lf ms\n\n", 1000.0 * elapsed / iteration);

    /*********************************************************
     * OpenCV version (CUFFT, CUBLAS, FAST_MATH, GPU arch 8.6)
     */
    elapsed = 0.;
    for (int i = 0; i < iteration; i++) {
        t.tic();
        matmult_opencv(m, n, k, &A[0], &B[0], &C[0]);
        elapsed += t.toc();
    }
    printf("OPENCV %s \n", validate(R, C, m, n) ? "PASSED" : "FAILED");
    printf("OPENCV %lf ms\n\n", 1000.0 * elapsed / iteration);

    /***************************************
     * SIMD version (SSE, AVX, AVX2, AVX512)
     */
    elapsed = 0.;
    for (int i = 0; i < iteration; i++) {
        t.tic();
        david::algorithms::math::matmult_simd(m, n, k, &A[0], &B[0], &C[0]);
        elapsed += t.toc();
    }
    printf("SIMD %s \n", validate(R, C, m, n) ? "PASSED" : "FAILED");
    printf("SIMD %lf ms\n\n", 1000.0 * elapsed / iteration);

    /*********************
     * CUDA version
     */
    // Warm up
    float _A[1], _B[1], _C[1];
    david::algorithms::math::matmult_cuda(1, 1, 1, _A, _B, _C, 1);
    elapsed = 0.;
    for (int i = 0; i < iteration; i++) {
        t.tic();
        /* The last argument
           '0' == global mem
           '1' == shared mem
           '2' == register
        */
        david::algorithms::math::matmult_cuda(m, n, k, &A[0], &B[0], &C[0], 1);
        elapsed += t.toc();
    }
    printf("CUDA %s \n", validate(R, C, m, n) ? "PASSED" : "FAILED");
    printf("CUDA %lf ms\n\n", 1000.0 * elapsed / iteration);

    return 0;
}
