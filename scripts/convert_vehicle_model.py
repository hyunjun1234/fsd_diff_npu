"""glTF/GLB 차량 모델을 3D 화면용 저폴리 메시(npz)로 바꾼다.

파이 CPU 렌더러(realityview/render_v15.py)는 삼각형을 하나씩 칠하므로 원본(수만~수십만 면)을 그대로 쓸 수 없다.
부품별로 면 수를 줄이고, 재질을 역할(도장, 유리, 타이어, 휠, 등화, 검은 부품)로 나눠 면마다 색을 붙여 저장한다.

실행(PC에서 한 번, trimesh와 fast-simplification 필요):
  pip install trimesh fast-simplification
  python scripts/convert_vehicle_model.py kia_carnival_2011.glb assets/vehicles/ego_carnival_2011.npz \
      --length 5.13 --paint 245,245,245

좌표: 결과는 차량 좌표(X 전방, Y 좌측, Z 위, m)이고 원점은 뒷바퀴 축 가운데 바닥이다.
원본은 Sketchfab 내보내기 기준(로컬 y 전방, z 위, x 오른쪽)이라고 보고 바꾼다(--axes로 바꿀 수 있음).
"""
from __future__ import annotations

import argparse
import re

import numpy as np

# 재질 이름(정규식) -> 역할. 위에서부터 처음 맞는 것을 쓴다. 맞는 것이 없으면 원본 색을 그대로 쓴다(trim)
ROLE_RULES = [
    ("drop", r"^Logo|untitledinttex|Meshpart1Mtl\.007|kcvl26211Mtl\.005"),  # 엠블럼, 실내 질감, 차체 안쪽 겹면(도장과 겹쳐 얼룩이 생김)
    ("paint", r"kcvl05411Mtl"),  # 차체 도장
    ("glass", r"^Windows|kcvl05011Mtl"),
    ("tire", r"^Tire"),
    ("rim", r"^Rims"),
    ("head", r"^Headlight|kcvl05311Mtl\.016"),
    ("tail", r"^Tailight|^Taillight|kcvl01511Mtl|^rEVERSE|^Blinker"),
]
ROLE_ID = {"trim": 0, "paint": 1, "glass": 2, "tire": 3, "rim": 4, "head": 5, "tail": 6}
# 역할별 색(BGR). paint는 --paint로 바꾼다
ROLE_BGR = {
    "glass": (52, 40, 34),
    "tire": (28, 28, 30),
    "rim": (150, 150, 155),
    "head": (235, 240, 245),
    "tail": (40, 30, 170),
}


def cluster(V, F, C, R, cell: float):
    """정점 묶기(vertex clustering)로 면을 더 줄인다.

    원본 부품은 경계가 많아 quadric 면 줄이기가 일정 수 아래로 내려가지 않는다. 그래서 cell(m) 격자 칸마다
    정점을 하나로 합치고(역할이 다른 부품끼리는 합치지 않음), 넓이가 0이 된 면과 겹친 면을 뺀다.
    """
    if cell <= 0:
        return V, F, C, R
    role_v = np.zeros(len(V), np.int64)
    role_v[F[:, 0]] = R
    key = np.c_[np.floor(V / cell).astype(np.int64), role_v]
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    newV = np.zeros((len(uniq), 3))
    np.add.at(newV, inv, V)
    newV /= np.bincount(inv, minlength=len(uniq))[:, None]
    nf = inv[F]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf, C, R = nf[ok], C[ok], R[ok]
    srt = np.sort(nf, axis=1)
    _, first = np.unique(srt, axis=0, return_index=True)
    first.sort()
    nf, C, R = nf[first], C[first], R[first]
    used, nf2 = np.unique(nf, return_inverse=True)
    return newV[used], nf2.reshape(-1, 3), C, R


def role_of(name: str) -> str:
    for role, pat in ROLE_RULES:
        if re.search(pat, name or ""):
            return role
    return "trim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--length", type=float, required=True, help="실제 차 길이(m). 원본 크기를 이 길이에 맞춘다")
    ap.add_argument("--rear-axle", type=float, default=None,
                    help="차 뒤끝에서 뒷바퀴 축까지(m). 없으면 타이어 메시로 찾는다")
    ap.add_argument("--paint", default="245,245,245", help="도장 색 R,G,B. 기본 흰색")
    ap.add_argument("--faces", type=int, default=24000, help="전체 목표 면 수")
    ap.add_argument("--faces-low", type=int, default=8000, help="먼 거리, 시점 이동 중에 쓸 목표 면 수")
    ap.add_argument("--aggression", type=int, default=7, help="면 줄이기 강도(0~10). 클수록 목표 면 수에 가깝게 줄인다")
    ap.add_argument("--cell", type=float, default=0.03, help="정점 묶기 격자 크기(m). 0이면 끔")
    ap.add_argument("--cell-low", type=float, default=0.07, help="낮은 LOD의 정점 묶기 격자 크기(m)")
    ap.add_argument("--min-area", type=float, default=0.002, help="이보다 작은(m2) 부품은 뺀다")
    args = ap.parse_args()

    import trimesh

    scene = trimesh.load(args.src, force="scene")
    parts = []  # (정점(N,3) 원본 좌표, 면, 역할, 원본 색 BGR)
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        mat = g.visual.material
        mname = getattr(mat, "name", "") or ""
        role = role_of(mname)
        if role == "drop":
            continue
        c = getattr(mat, "baseColorFactor", None)
        rgb = np.array(c[:3] if c is not None else (128, 128, 128), dtype=np.float64)
        g = trimesh.Trimesh(vertices=np.asarray(g.vertices), faces=np.asarray(g.faces), process=False)
        g.apply_transform(T)
        # glTF는 법선과 UV 때문에 같은 자리 정점을 여러 개로 나눠 둔다. 합치지 않으면 면 줄이기가 이음매를 찢는다
        g.merge_vertices()
        parts.append((g, role, (int(rgb[2]), int(rgb[1]), int(rgb[0])), mname))

    allv = np.vstack([p[0].vertices for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    ext = hi - lo
    # 축 찾기: 가장 긴 축 = 길이, 가장 짧은 축 중 ... 장면 그래프를 거친 뒤 Y가 위인 glTF 관례를 따른다
    up = 1  # glTF는 Y 위
    length_ax = int(np.argmax([ext[0], -1, ext[2]]))
    width_ax = 2 if length_ax == 0 else 0
    k = args.length / ext[length_ax]
    # 앞쪽 방향: 전조등이 있는 쪽
    head = [p[0].vertices for p in parts if p[1] == "head"]
    front_sign = 1.0
    if head:
        hc = np.vstack(head)[:, length_ax].mean()
        front_sign = 1.0 if hc > (lo[length_ax] + hi[length_ax]) / 2 else -1.0

    def to_vehicle(v):
        X = front_sign * (v[:, length_ax] - (lo[length_ax] + hi[length_ax]) / 2) * k
        Z = (v[:, up] - lo[up]) * k
        # 오른손 좌표: (X 전방, Y 좌측, Z 위). glTF는 (x, y위, z) 오른손이라 Y = -(전방 x 위)로 구한다
        f = np.zeros(3)
        f[length_ax] = front_sign
        u = np.zeros(3)
        u[up] = 1.0
        left = np.cross(u, f)  # 위 x 전방 = 왼쪽
        Y = (v - (lo + hi) / 2) @ left * k
        return np.stack([X, Y, Z], 1)

    # 뒷바퀴 축: 타이어 정점 중 뒤쪽 절반의 X 평균
    if args.rear_axle is None:
        tires = [to_vehicle(p[0].vertices) for p in parts if p[1] == "tire"]
        tv = np.vstack(tires)
        rear = tv[tv[:, 0] < 0, 0]
        x_axle = float((rear.min() + rear.max()) / 2)
    else:
        x_axle = -args.length / 2 + args.rear_axle
    paint = tuple(int(c) for c in args.paint.split(","))[::-1]

    total_faces = sum(len(p[0].faces) for p in parts if p[0].area * k * k >= args.min_area)
    out = {}
    for tag, budget in (("", args.faces), ("_low", args.faces_low)):
        V, F, C, R = [], [], [], []
        nv = 0
        for g, role, bgr, mname in parts:
            area = g.area * k * k
            if area < args.min_area:
                continue
            # 부품마다 같은 비율로 줄인다(넓이 비례로 나누면 잘게 쪼갠 차체 패널에 구멍이 난다)
            target = max(8, int(len(g.faces) * budget / total_faces))
            m = g
            if len(m.faces) > target:
                try:
                    m = m.simplify_quadric_decimation(face_count=target, aggression=args.aggression)
                except Exception as e:  # 줄이지 못한 부품은 그대로 쓰고 알린다
                    print(f"면 줄이기 실패({mname}, {len(m.faces)}면): {e}")
            if len(m.faces) == 0:
                continue
            v = to_vehicle(m.vertices)
            v[:, 0] -= x_axle
            col = paint if role == "paint" else ROLE_BGR.get(role, bgr)
            if role == "trim":
                # 원본이 거의 검은색인 부품이 많다. 화면에서 형체가 보이게 조금 밝힌다
                col = tuple(int(min(255, 40 + 0.8 * c)) for c in col)
            V.append(v)
            F.append(np.asarray(m.faces) + nv)
            C.append(np.tile(np.array(col, np.uint8), (len(m.faces), 1)))
            R.append(np.full(len(m.faces), ROLE_ID[role], np.uint8))
            nv += len(v)
        Vv, Ff, Cc, Rr = cluster(np.vstack(V), np.vstack(F), np.vstack(C), np.concatenate(R),
                                 args.cell if tag == "" else args.cell_low)
        out["V" + tag] = Vv.astype(np.float32)
        out["F" + tag] = Ff.astype(np.int32)
        out["C" + tag] = Cc
        out["R" + tag] = Rr
        nv = len(Vv)
        print(f"lod{tag or '_high'}: 정점 {nv}, 면 {len(out['F' + tag])}")
    v = out["V"]
    print("크기(m): 길이 %.2f 폭 %.2f 높이 %.2f, 뒷축에서 뒤끝 %.2f" % (
        np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2]), -v[:, 0].min()))
    np.savez_compressed(args.dst, **out)


if __name__ == "__main__":
    main()
