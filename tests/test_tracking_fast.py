from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.bev import box_center, camera_ground_axis, project_detections  # noqa: E402
from realityview.config import load_config  # noqa: E402
from realityview.multicam import FrameSyncer, ReconnectingSource, SyncedSources, UsbCameraSource  # noqa: E402
from realityview.sim import LANE_W, SimObject, SimWorld, default_objects  # noqa: E402
from realityview.tracker import BevTracker, hungarian  # noqa: E402
from realityview.types import GroundObject  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _cfg(name: str) -> str:
    return os.path.join(ROOT, "configs", name)


def _car(x: float, y: float) -> GroundObject:
    return GroundObject(label="car", score=0.9, xy=np.array([x, y], float), size=(4.5, 1.9), cam_index=0, distance=0.0)


# ---------------------------------------------------------------------------
# 1. sim 재현: 마주 오는 차(상대 25 m/s)의 ID 바뀜
# ---------------------------------------------------------------------------
def test_oncoming_car_sim_repro():
    """카메라 2대, 15fps, 75프레임(마주 오는 차가 두 번 지나감). basic은 이 차에 트랙이 여러 개 붙고,
    robust(칼만 판)는 첫 번째 지나감에 2개 이하(같은 차로 차를 뚫고 지나가는 순간의 맞바뀜만), 두 번째에 1개다."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eval_tracking as ev

    res = {}
    for trk in ("basic", "robust"):
        m, _ = ev.run_sim_sync(_cfg("two_cam.yaml"), 15, 75, 0, trk, 0, [0, 0], [0, 0], [0, 0], 25, "hold", False, True)
        res[trk] = m
    basic, robust = res["basic"], res["robust"]
    assert len(basic._ids[(6, 0)]) >= 4
    assert len(robust._ids[(6, 0)]) <= 2
    assert len(robust._ids[(6, 1)]) == 1
    assert robust.switches < basic.switches


# ---------------------------------------------------------------------------
# 2. 추적기 단위 시험
# ---------------------------------------------------------------------------
def test_crossing_tracks_keep_ids():
    """같은 차로에서 서로 스쳐 가는 두 차(상대 속도 27 m/s). 겹친 트랙 정리(e)가 빠른 트랙을 지우면 안 된다."""
    tr = BevTracker(mode="robust")
    dt = 1.0 / 15
    ids_fast, ids_slow = set(), set()
    for n in range(40):
        t = n * dt
        fast = _car(30.0 - 25.0 * t, 3.5)
        slow = _car(8.0 + 2.0 * t, 3.5)
        out = tr.update([fast, slow], dt)
        for o in out:
            if n > 3:
                (ids_fast if o.vel[0] < -5 else ids_slow).add(o.tid)
    assert len(ids_fast) == 1, ids_fast
    assert len(ids_slow) == 1, ids_slow


def test_fast_track_survives_blind_gap():
    """속도가 자리 잡은 빠른 트랙은 7프레임 동안 안 보여도(앞 카메라와 뒤 카메라 사이) 같은 ID로 이어진다."""
    tr = BevTracker(mode="robust")
    dt = 1.0 / 15
    tid = None
    for n in range(30):
        x = 20.0 - 25.0 * n * dt
        seen = not (14 <= n < 21)
        out = tr.update([_car(x, 3.5)] if seen else [], dt)
        if n == 13:
            tid = out[0].tid
        if n == 22:
            assert [o.tid for o in out] == [tid]


def test_new_track_lateral_velocity_is_damped():
    """처음 두 측정 사이 옆으로 1.5m 튄 잡음이 옆 속도로 크게 바뀌지 않는다(옆 속도 처음 불확실성 3 m/s)."""
    tr = BevTracker(mode="robust")
    tr.update([_car(-4.4, 3.2)], 1 / 15)
    tr.update([_car(-6.1, 4.7)], 1 / 15)
    assert abs(tr.tracks[0].vel[1]) < 5.0


def test_side_lane_car_not_swapped():
    """나란히 가는 옆 차선 차(3.5m 옆)와 ID가 바뀌지 않는다(옆 방향 gate)."""
    tr = BevTracker(mode="robust")
    rng = np.random.default_rng(0)
    seen = {}
    for n in range(60):
        a = _car(20.0 + rng.normal(0, 1.0), 0.0 + rng.normal(0, 0.3))
        b = _car(20.0 + rng.normal(0, 1.0), 3.5 + rng.normal(0, 0.3))
        for o in tr.update([a, b], 1 / 15):
            if n > 3:
                seen.setdefault(round(o.xy[1] / 3.5), set()).add(o.tid)
    assert all(len(v) == 1 for v in seen.values()), seen


def test_basic_mode_unchanged_default():
    tr = BevTracker()
    assert tr.mode == "basic"
    tr.update([_car(10, 0)], 0.1)
    assert tr.tracks[0].cov is None  # basic은 칼만 상태를 쓰지 않는다


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n, m = (int(v) for v in rng.integers(1, 5, 2))
        C = rng.normal(size=(n, m)) * 10
        got = hungarian(C)
        assert len(got) == min(n, m)
        assert len({r for r, _ in got}) == len(got) and len({c for _, c in got}) == len(got)
        best = None
        if n <= m:
            for perm in itertools.permutations(range(m), n):
                v = sum(C[i, perm[i]] for i in range(n))
                best = v if best is None else min(best, v)
        else:
            for perm in itertools.permutations(range(n), m):
                v = sum(C[perm[j], j] for j in range(m))
                best = v if best is None else min(best, v)
        assert abs(sum(C[r, c] for r, c in got) - best) < 1e-9
    assert hungarian(np.zeros((0, 3))) == []


# ---------------------------------------------------------------------------
# 3. 차 중심 추정(vehicle_center="box")
# ---------------------------------------------------------------------------
def test_box_center_straight_ahead_equals_half_length():
    g = np.array([10.0, 0.0])
    c = box_center(g, np.array([0.0, 0.0]), np.array([1.0, 0.0]), 4.5, 1.9)
    assert np.allclose(c, [12.25, 0.0])


def test_box_center_reduces_side_camera_error():
    """카메라 4대 sim(잡음 없음)에서 측면 카메라의 차 중심 오차가 ray 방식보다 작다."""
    cams, _, _ = load_config(_cfg("four_cam.yaml"))
    objs = default_objects()
    objs.append(SimObject("car", 40.0, LANE_W, -25.0, 0.0, 4.5, 1.9, 1.5, (90, 160, 90)))
    w = SimWorld(cams, objects=objs)
    err = {"ray": [], "box": []}
    for n in range(0, 225, 5):
        w.set_time(n / 15)
        for k in (2, 3):  # 왼쪽, 오른쪽 측면 카메라
            _, dets, ids, _ = w.render_camera(k, ground=False)
            for d, i in zip(dets, ids):
                o = w.objects[i]
                if o.label == "person":
                    continue
                for mode in err:
                    p = project_detections(k, cams[k], [d], max_range=40, vehicle_center=mode)
                    if p:
                        err[mode].append(float(np.linalg.norm(p[0].xy - [o.x, o.y])))
    assert len(err["box"]) > 20
    assert np.mean(err["box"]) < 0.6 * np.mean(err["ray"])
    assert camera_ground_axis(cams[0]) is not None
    with pytest.raises(ValueError):
        project_detections(0, cams[0], [], vehicle_center="nope")


# ---------------------------------------------------------------------------
# 4. 카메라 끊김과 복구
# ---------------------------------------------------------------------------
class _FakeCap:
    """가짜 VideoCapture. fail_after장 뒤로는 read()가 실패한다(케이블이 빠진 장치)."""

    def __init__(self, fail_after: int | None):
        self.n = 0
        self.fail_after = fail_after

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            time.sleep(0.01)
            return False, None
        time.sleep(0.005)
        return True, np.full((48, 64, 3), 100, np.uint8)

    def release(self):
        pass


class _Ref:
    """기준 카메라 흉내: 늘 영상을 주고 시각은 지금."""

    def __init__(self):
        self.last_timestamp_ns = None

    def read(self):
        time.sleep(0.01)
        self.last_timestamp_ns = time.monotonic_ns()
        return np.zeros((48, 64, 3), np.uint8)

    def close(self):
        pass


def test_reconnect_usb_source_through_synced_sources():
    """USB 카메라가 20장 뒤 끊기고, 다시 열기가 두 번 실패한 뒤 세 번째에 살아난다.
    끊긴 동안 그 칸은 lost(NO SIGNAL), 다른 카메라는 계속 읽히고, read()는 오래 막히지 않는다."""
    opens = {"n": 0}

    def open_usb():
        opens["n"] += 1
        if opens["n"] == 1:
            cap = _FakeCap(fail_after=20)
        elif opens["n"] in (2, 3):
            time.sleep(0.2)  # 느린 열기: 메인 루프를 막으면 안 된다
            raise IOError("아직 장치가 없습니다")
        else:
            cap = _FakeCap(fail_after=None)
        return UsbCameraSource("/dev/fake", 64, 48, capture_factory=lambda d: cap, wait_s=0.3)

    usb = ReconnectingSource(open_usb, name="left", retry_s=0.05)
    sy = FrameSyncer([(64, 48), (64, 48)], ["front", "left"], tolerance_ms=50)
    reader = SyncedSources([_Ref(), usb], sy)
    statuses, slow = [], 0
    t_end = time.monotonic() + 10.0
    try:
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            frames = reader.read()
            dt = time.monotonic() - t0
            assert frames is not None and len(frames) == 2
            statuses.append(reader.last.status[1])
            if usb.signal_lost and dt > 0.15:
                slow += 1
            if "lost" in statuses and statuses[-1] == "ok" and usb.reopens >= 1:
                break
    finally:
        usb.close()
    assert "lost" in statuses
    assert statuses[-1] == "ok"
    st = usb.stats()
    assert st["drops"] == 1 and st["reopens"] == 1 and st["retries"] >= 3
    assert slow <= 1  # 끊김을 알아챈 첫 read(wait_s 0.3초) 말고는 막히지 않는다
    assert sy.summary()["cameras"]["left"]["lost"] >= 1


def test_syncer_lost_clears_hold():
    sy = FrameSyncer([(32, 24), (32, 24)], ["a", "b"], tolerance_ms=10, on_miss="hold")
    sy.push(0, np.zeros((24, 32, 3), np.uint8), 1_000_000_000)
    sy.push(1, np.full((24, 32, 3), 7, np.uint8), 1_000_000_000)
    b = sy.bundle(1_000_000_000)
    assert b.status == ["ok", "ok"]
    sy.push(0, np.zeros((24, 32, 3), np.uint8), 1_100_000_000)
    b = sy.bundle(1_100_000_000, lost={1})
    assert b.status == ["ok", "lost"] and b.offsets_ms[1] is None
    assert b.frames[1].shape == (24, 32, 3) and b.frames[1].max() > 40  # NO SIGNAL 글자
    sy.push(0, np.zeros((24, 32, 3), np.uint8), 1_200_000_000)
    b = sy.bundle(1_200_000_000)
    assert b.status[1] == "blank"  # 끊기기 전 프레임을 hold로 다시 쓰지 않는다


def test_reference_camera_lost_keeps_others():
    class Flaky:
        def __init__(self):
            self.signal_lost = True
            self.last_timestamp_ns = None

        def read(self):
            return None

    sy = FrameSyncer([(32, 24), (32, 24)], ["front", "rear"], tolerance_ms=50)
    reader = SyncedSources([Flaky(), _Ref()], sy, lost_wait_s=0.0)
    frames = reader.read()
    assert frames is not None
    assert reader.last.status == ["lost", "ok"]


def test_reconnecting_source_forwards_attributes():
    class Src:
        def __init__(self):
            self.want_timestamp = False
            self.last_timestamp_ns = 5

        def read(self):
            return np.zeros((2, 2, 3), np.uint8)

        def close(self):
            pass

    r = ReconnectingSource(Src, name="x")
    r.want_timestamp = True
    assert r.src.want_timestamp is True
    assert r.last_timestamp_ns == 5
    assert r.read() is not None and not r.signal_lost
    r.close()
