//≥™¿Ã∫Í ƒ⁄µÂ øπΩ√

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <vector>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
#include <iostream>

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

// My implementation begins
template <typename T>
union VecPtr {
    const float* data;
    T* p;
};

static inline __m256 reduce_ma_pack8(int k, int n, const float* a, const float* b) {
    __m256 buf;
    memcpy(&buf, b, sizeof(__m256));
    __m256 x = _mm256_mul_ps(_mm256_broadcast_ss(&a[0]), buf);
    int offset = n;

    for (int i = 1; i < k; ++i) {
        memcpy(&buf, b + offset, sizeof(__m256));
        x = _mm256_add_ps(x, _mm256_mul_ps(_mm256_broadcast_ss(&a[i]), buf));
        offset += n;
    }

    return x;
}

static inline __m128 reduce_ma_pack4(int k, int n, const float* a, const float* b) {
    __m128 buf;
    memcpy(&buf, b, sizeof(__m128));
    __m128 x = _mm_mul_ps(_mm_broadcast_ss(&a[0]), buf);
    int offset = n;

    for (int i = 1; i < k; ++i) {
        memcpy(&buf, b + offset, sizeof(__m128));
        x = _mm_add_ps(x, _mm_mul_ps(_mm_broadcast_ss(&a[i]), buf));
        offset += n;
    }

    return x;
}

static inline float work_hard(int k, int n, const float* a, const float* b) {
    float x = .0;
    int irow = 0;

    for (int i = 0; i < k; ++i) {
        x += a[i] * b[irow];
        irow += n;
    }

    return x;
}

static inline void compute_row(int k, int n, int n_quant8, int n_quant4, int n_remainder, const float* a_row, const float* b, float* c_row) {
    int stride = 0;

    for (int i = 0; i < n_quant8; ++i) {
        __m256 result = reduce_ma_pack8(k, n, a_row, b + stride);

        // copy the result to c
        VecPtr<__m256> tmp;
        tmp.p = &result;
        memcpy(c_row + stride, tmp.data, 8 * sizeof(float));

        // update the index
        stride += 8;
    }

    for (int i = 0; i < n_quant4; ++i) {
        __m128 result = reduce_ma_pack4(k, n, a_row, b + stride);

        // copy the result to c
        VecPtr<__m128> tmp;
        tmp.p = &result;
        memcpy(c_row + stride, tmp.data, 4 * sizeof(float));

        // update the index
        stride += 4;
    }

    for (int i = 0; i < n_remainder; ++i) {
        float result = work_hard(k, n, a_row, b + stride);
        c_row[stride] = result;
        stride += 1;
    }
}

void matmult_opt(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c) {

    // column
    int n_quant8 = n / 8;
    int n_quant4 = (n - n_quant8 * 8) / 4;
    int n_remainder = n - n_quant8 * 8 - n_quant4 * 4;

    // row
    int m_quant8 = m / 8; // 0
    int m_quant4 = (m - m_quant8 * 8) / 4; // 1
    int m_remainder = m - m_quant8 * 8 - m_quant4 * 4; // 1

    // std::cout << "(" << m << " x " << k << "), (" << k << " x " << n << ")" << std::endl;
    // std::cout << "n_quant8: " << n_quant8 << std::endl;
    // std::cout << "n_quant4: " << n_quant4 << std::endl;
    // std::cout << "n_remainder: " << n_remainder << std::endl;
    // std::cout << std::endl;

    int sa = 0, sc = 0; // strides

    for (int i = 0; i < m_quant8; ++i) {
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }

    for (int i = 0; i < m_quant4; ++i) {
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }

    for (int i = 0; i < m_remainder; ++i) {
        compute_row(k, n, n_quant8, n_quant4, n_remainder, mat_a+sa, mat_b, mat_c+sc); sa+=k; sc+=n;
    }
}
// My implementation ends

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
    // srand(0); // see if it reproduce the result of reference (naive one)
    mat.resize(n * m);
    for (int i=0; i < mat.size(); i++) mat[i] = rand() % 100;
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
    std::vector<float> mat_a;
    std::vector<float> mat_b;
    std::vector<float> mat_c;

    int m = 10;
    int k = 10;
    int n = 10;

    genmat(m, k, mat_a);
    genmat(k, n, mat_b);
    genmat(m, n, mat_c);

    Timer t;
    double elapsed=0;
    const int iteration = 10000;
    for (int i=0; i<iteration; i++)
    {
        t.tic();
        //matmult(m, n, k, &mat_a[0], &mat_b[0], &mat_c[0]);
        matmult_opt(m, n, k, &mat_a[0], &mat_b[0], &mat_c[0]); // my implementation
        //matmult_opencv(m, n, k, &mat_a[0], &mat_b[0], &mat_c[0]);
        elapsed += t.toc();
    }

    dumpmat(m, k, mat_a); printf("\n");
    dumpmat(m, n, mat_c); printf("\n");
    printf("%lf ms\n", 1000.0 * elapsed / iteration);
    return 0;
}
