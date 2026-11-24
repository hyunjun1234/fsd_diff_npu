#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ===== 빌드 정보 =====
 * 파이썬 쪽에서 OpenMP가 실제로 켜졌는지 확인할 때 부릅니다.
 * 1이면 -fopenmp로 빌드되어 여러 스레드를 쓸 수 있고, 0이면 항상 한 스레드입니다. */
int bevpool_has_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/* 요청한 스레드 수로 실제 몇 개가 돌았는지 돌려줍니다(측정 기록용). */
int bevpool_threads_used(int nthreads)
{
#ifdef _OPENMP
    int used = 1;
    if (nthreads < 1) nthreads = 1;
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp single
        used = omp_get_num_threads();
    }
    return used;
#else
    (void)nthreads;
    return 1;
#endif
}

/* ===== BEVPool 누적 =====
 * 미리 계산한 계획(plan)에 따라 깊이 확률 x 이미지 특징을 BEV 셀에 더합니다.
 *
 * 입력
 *   depth      : 깊이 확률을 1차원으로 편 배열, 길이 N*D*H*W (float32)
 *   feat_t     : 특징을 (N*H*W, C) 순서로 바꾼 배열 (float32, 한 픽셀의 C개 채널이 붙어 있음)
 *   C          : 채널 수
 *   depth_idx  : 격자 안에 들어온 점마다 depth 배열의 위치 (BEV 셀 순서로 정렬됨)
 *   feat_idx   : 같은 점의 feat_t 행 번호
 *   iv_start   : 셀(구간)마다 첫 점의 위치
 *   iv_len     : 셀마다 점 개수
 *   iv_cell    : 셀마다 BEV 평면 인덱스 (ix * Y + iy)
 *   n_iv       : 점이 하나라도 있는 셀 수
 *   XY         : BEV 셀 총 개수 (X * Y)
 *   out        : 결과 (C, XY), 호출 전에 0으로 채워 둬야 함
 *   nthreads   : OpenMP 스레드 수 (OpenMP 없이 빌드되면 무시)
 *
 * 셀마다 한 스레드가 맡아 끝까지 더하므로 같은 셀에 두 스레드가 동시에 쓰는 일이 없습니다.
 * 그래서 원자적 덧셈(atomic)이 필요 없습니다(BEVPoolv2와 같은 방식). */
void bevpool_accumulate(const float *depth,
                        const float *feat_t,
                        int C,
                        const int32_t *depth_idx,
                        const int32_t *feat_idx,
                        const int32_t *iv_start,
                        const int32_t *iv_len,
                        const int32_t *iv_cell,
                        int n_iv,
                        int XY,
                        float *out,
                        int nthreads)
{
    if (nthreads < 1) nthreads = 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
#endif
    for (int i = 0; i < n_iv; ++i) {
        /* 한 셀의 채널별 합을 스택에 모읍니다. C가 크면 나눠서 처리합니다. */
        float acc[256];
        const int s = iv_start[i];
        const int e = s + iv_len[i];
        const int cell = iv_cell[i];

        for (int c0 = 0; c0 < C; c0 += 256) {
            const int cn = (C - c0) < 256 ? (C - c0) : 256;
            memset(acc, 0, sizeof(float) * (size_t)cn);

            /* 이 셀에 떨어진 frustum 점을 하나씩 더합니다. */
            for (int p = s; p < e; ++p) {
                const float w = depth[depth_idx[p]];
                const float *f = feat_t + (size_t)feat_idx[p] * (size_t)C + c0;
                for (int c = 0; c < cn; ++c) {
                    acc[c] += w * f[c];
                }
            }

            /* (C, XY) 순서로 결과에 씁니다. */
            for (int c = 0; c < cn; ++c) {
                out[(size_t)(c0 + c) * (size_t)XY + (size_t)cell] = acc[c];
            }
        }
    }
}
