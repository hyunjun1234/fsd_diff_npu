#include <stdint.h>
#include <math.h>

/* ===== 삼각형 z-버퍼 래스터라이저 =====
 * render_v15.py가 3D 물체(자차, 차, 사람, 신호등)를 칠할 때 씁니다.
 * 파이썬에서 면마다 cv2.fillConvexPoly를 부르면 면 수천 개에서 수십 ms가 들어서 C로 옮겼습니다.
 *
 * 입력
 *   uv    : 정점 화면 좌표 (nv x 2, 화소, 소수 가능)
 *   iz    : 정점 깊이의 역수 1/z (nv). 화면 공간에서 선형 보간해도 원근이 맞는 값입니다
 *   tris  : 삼각형 정점 번호 (nt x 3)
 *   col   : 면 색 BGR (nt x 3). 조명과 안개는 파이썬에서 미리 곱해 둡니다
 *   alpha : 면 불투명도 0~255 (nt). 255면 덮어쓰고 z도 씁니다. 그보다 작으면 섞기만 하고 z는 쓰지 않습니다
 * 출력(그 자리에서 고침)
 *   img   : H x W x 3 uint8
 *   zbuf  : H x W float (1/z, 클수록 가깝다). 0이면 비어 있음
 *   mask  : H x W uint8 또는 NULL. 칠한 화소에 255를 씁니다(자차 그림 캐시용)
 * 화소 중심(x+0.5, y+0.5)이 삼각형 안에 있으면 칠합니다(위쪽, 왼쪽 경계 규칙은 단순화). 뒷면 제거는 파이썬에서 합니다.
 */

/* 한 줄(py)에서 세 경계 함수가 모두 0 이상인 x 구간 [*xa, *xb]를 구한다.
 * 경계 함수 w_i(x) = A_i * x + B_i 가 x에 선형이라 구간을 바로 풀 수 있다(빈 화소를 돌지 않는다). */
static int row_span(const float *A, const float *B, int minx, int maxx, int *xa, int *xb)
{
    float lo = minx + 0.5f, hi = maxx + 0.5f;
    for (int i = 0; i < 3; i++) {
        if (A[i] > 1e-12f) {
            float r = -B[i] / A[i];
            if (r > lo) lo = r;
        } else if (A[i] < -1e-12f) {
            float r = -B[i] / A[i];
            if (r < hi) hi = r;
        } else if (B[i] < 0) {
            return 0;
        }
    }
    if (hi < lo) return 0;
    int a = (int)ceilf(lo - 0.5f), b = (int)floorf(hi - 0.5f);
    if (a < minx) a = minx;
    if (b > maxx) b = maxx;
    if (b < a) return 0;
    *xa = a;
    *xb = b;
    return 1;
}

int raster_tris(const float *uv, const float *iz, int nv,
                const int32_t *tris, int nt,
                const uint8_t *col, const uint8_t *alpha,
                uint8_t *img, float *zbuf, uint8_t *mask, int W, int H)
{
    int drawn = 0;
    for (int t = 0; t < nt; t++) {
        int i0 = tris[3 * t], i1 = tris[3 * t + 1], i2 = tris[3 * t + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= nv || i1 >= nv || i2 >= nv) continue;
        float x0 = uv[2 * i0], y0 = uv[2 * i0 + 1];
        float x1 = uv[2 * i1], y1 = uv[2 * i1 + 1];
        float x2 = uv[2 * i2], y2 = uv[2 * i2 + 1];
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (fabsf(area) < 1e-6f) continue;
        float fminx = fminf(x0, fminf(x1, x2)), fmaxx = fmaxf(x0, fmaxf(x1, x2));
        float fminy = fminf(y0, fminf(y1, y2)), fmaxy = fmaxf(y0, fmaxf(y1, y2));
        if (fmaxx < 0 || fmaxy < 0 || fminx >= W || fminy >= H) continue;
        int minx = (int)floorf(fminx), maxx = (int)ceilf(fmaxx);
        int miny = (int)floorf(fminy), maxy = (int)ceilf(fmaxy);
        if (minx < 0) minx = 0;
        if (miny < 0) miny = 0;
        if (maxx > W - 1) maxx = W - 1;
        if (maxy > H - 1) maxy = H - 1;
        float inv = 1.0f / area;
        float z0 = iz[i0], z1 = iz[i1], z2 = iz[i2];
        uint8_t b = col[3 * t], g = col[3 * t + 1], r = col[3 * t + 2];
        int a = alpha ? alpha[t] : 255;
        int opaque = a >= 255;
        /* w0 = ((x1-px)(y2-py) - (x2-px)(y1-py)) * inv 를 px에 대해 A*px + B 꼴로 */
        for (int y = miny; y <= maxy; y++) {
            float py = y + 0.5f;
            float A[3], B[3];
            A[0] = (-(y2 - py) + (y1 - py)) * inv;
            B[0] = (x1 * (y2 - py) - x2 * (y1 - py)) * inv;
            A[1] = (-(y0 - py) + (y2 - py)) * inv;
            B[1] = (x2 * (y0 - py) - x0 * (y2 - py)) * inv;
            A[2] = -A[0] - A[1];
            B[2] = 1.0f - B[0] - B[1];
            int xa, xb;
            if (!row_span(A, B, minx, maxx, &xa, &xb)) continue;
            for (int x = xa; x <= xb; x++) {
                float px = x + 0.5f;
                float w0 = A[0] * px + B[0];
                float w1 = A[1] * px + B[1];
                float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
                float z = w0 * z0 + w1 * z1 + w2 * z2;
                int k = y * W + x;
                if (z <= zbuf[k]) continue;
                uint8_t *p = img + 3 * k;
                if (opaque) {
                    zbuf[k] = z;
                    p[0] = b; p[1] = g; p[2] = r;
                    if (mask) mask[k] = 255;
                } else {
                    p[0] = (uint8_t)((p[0] * (255 - a) + b * a) / 255);
                    p[1] = (uint8_t)((p[1] * (255 - a) + g * a) / 255);
                    p[2] = (uint8_t)((p[2] * (255 - a) + r * a) / 255);
                    if (mask) mask[k] = 255;
                }
                drawn++;
            }
        }
    }
    return drawn;
}

/* ===== 꼭짓점 색 보간판(Gouraud) =====
 * raster_tris와 같지만 col이 삼각형 꼭짓점마다 색(nt x 3 x 3, BGR)이다. 화면 공간에서 색을 선형 보간해
 * 저폴리 차체의 각진 면이 부드럽게 보이게 한다. 모두 불투명으로 그린다. */
int raster_tris_vc(const float *uv, const float *iz, int nv,
                   const int32_t *tris, int nt, const uint8_t *col,
                   uint8_t *img, float *zbuf, uint8_t *mask, int W, int H)
{
    int drawn = 0;
    for (int t = 0; t < nt; t++) {
        int i0 = tris[3 * t], i1 = tris[3 * t + 1], i2 = tris[3 * t + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= nv || i1 >= nv || i2 >= nv) continue;
        float x0 = uv[2 * i0], y0 = uv[2 * i0 + 1];
        float x1 = uv[2 * i1], y1 = uv[2 * i1 + 1];
        float x2 = uv[2 * i2], y2 = uv[2 * i2 + 1];
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (fabsf(area) < 1e-6f) continue;
        float fminx = fminf(x0, fminf(x1, x2)), fmaxx = fmaxf(x0, fmaxf(x1, x2));
        float fminy = fminf(y0, fminf(y1, y2)), fmaxy = fmaxf(y0, fmaxf(y1, y2));
        if (fmaxx < 0 || fmaxy < 0 || fminx >= W || fminy >= H) continue;
        int minx = (int)floorf(fminx), maxx = (int)ceilf(fmaxx);
        int miny = (int)floorf(fminy), maxy = (int)ceilf(fmaxy);
        if (minx < 0) minx = 0;
        if (miny < 0) miny = 0;
        if (maxx > W - 1) maxx = W - 1;
        if (maxy > H - 1) maxy = H - 1;
        float inv = 1.0f / area;
        float z0 = iz[i0], z1 = iz[i1], z2 = iz[i2];
        const uint8_t *c0 = col + 9 * t, *c1 = c0 + 3, *c2 = c0 + 6;
        /* w0 = ((x1-px)(y2-py) - (x2-px)(y1-py)) * inv 를 px에 대해 A*px + B 꼴로 */
        for (int y = miny; y <= maxy; y++) {
            float py = y + 0.5f;
            float A[3], B[3];
            A[0] = (-(y2 - py) + (y1 - py)) * inv;
            B[0] = (x1 * (y2 - py) - x2 * (y1 - py)) * inv;
            A[1] = (-(y0 - py) + (y2 - py)) * inv;
            B[1] = (x2 * (y0 - py) - x0 * (y2 - py)) * inv;
            A[2] = -A[0] - A[1];
            B[2] = 1.0f - B[0] - B[1];
            int xa, xb;
            if (!row_span(A, B, minx, maxx, &xa, &xb)) continue;
            for (int x = xa; x <= xb; x++) {
                float px = x + 0.5f;
                float w0 = A[0] * px + B[0];
                float w1 = A[1] * px + B[1];
                float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
                float z = w0 * z0 + w1 * z1 + w2 * z2;
                int k = y * W + x;
                if (z <= zbuf[k]) continue;
                zbuf[k] = z;
                uint8_t *p = img + 3 * k;
                for (int ch = 0; ch < 3; ch++) {
                    float v = w0 * c0[ch] + w1 * c1[ch] + w2 * c2[ch] + 0.5f;
                    p[ch] = (uint8_t)(v > 255.0f ? 255.0f : v);
                }
                if (mask) mask[k] = 255;
                drawn++;
            }
        }
    }
    return drawn;
}
