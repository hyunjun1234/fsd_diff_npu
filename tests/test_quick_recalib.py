from __future__ import annotations

import dataclasses
import os
import shutil
import warnings

import cv2
import numpy as np
import pytest
import yaml

from realityview import calib as C
from realityview.bev import IpmCompositor
from realityview.calib_monitor import CalibMonitor, CalibMonitorConfig
from realityview.calib_overlap import overlap_pairs, overlap_score
from realityview.config import load_config
from realityview.pipeline import RealityViewPipeline
from realityview.sim import SimBackend, SimWorld

ROOT = os.path.join(os.path.dirname(__file__), "..")
CUR = os.path.join(ROOT, "configs", "two_cam_current.yaml")
FOUR = os.path.join(ROOT, "configs", "four_cam.yaml")

# 빠른 재보정 도구 시험(바닥 마커 재보정, 주행 중 보정 점검, 카메라 간 겹침 확인).
# 합성 장면에서 카메라 각도를 일부러 1~3도 틀어 영상을 그리고(실제 카메라), 처리는 설정 값(틀어지기 전)으로 해서
# 각 도구가 틀어짐을 찾아내는지 본다. 틀지 않은 경우도 꼭 같이 돌려, 늘 경고만 내는 도구가 통과하지 못하게 한다.
# 거리 정확도는 카메라 보정 전이라 판정하지 않는다(합성 장면 안의 기하만 본다).

# 바닥 마커 배치(번호, X, Y). 전방은 카메라 앞 2~4m, 후방은 카메라 뒤 2~4m
FRONT_MARKERS = [(0, 3.8, 1.2), (1, 3.8, -1.2), (2, 5.0, 1.6), (3, 5.0, -1.6), (4, 4.4, 0.0), (5, 6.0, 0.3)]
REAR_MARKERS = [(0, -3.0, 0.8), (1, -3.0, -0.8), (2, -4.0, 1.2), (3, -4.0, -1.2), (4, -3.5, 0.0), (5, -4.8, 0.3)]


@pytest.fixture(autouse=True)
def _quiet():
    # 어안 가장자리 화소의 0 나누기 경고(결과는 valid로 걸러진다)를 시험 출력에서 숨긴다
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _tilt(cam, dyaw=0.0, dpitch=0.0, droll=0.0):
    return dataclasses.replace(cam, yaw_deg=cam.yaw_deg + dyaw, pitch_deg=cam.pitch_deg + dpitch,
                               roll_deg=cam.roll_deg + droll)


def _marker_frame(true_cam, layout, kind="aruco"):
    """실제(틀어진) 카메라로 빈 도로와 바닥 마커(한 변 0.5m, 번호마다 15도씩 돌림)를 그린 영상."""
    img = SimWorld([true_cam], objects=[]).render()[0]
    return C.render_floor_markers(true_cam, img, [(i, X, Y, 0.5, 15.0 * i) for i, X, Y in layout], kind=kind)


def _detect(nom, img, layout, kind):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lay = {i: (X, Y) for i, X, Y in layout}
    det = C.detect_aruco(g, nom) if kind == "aruco" else C.detect_circles(g, nom, lay, 60.0)
    ids = sorted(det)
    return np.array([det[i] for i in ids]), np.array([lay[i] for i in ids]), ids


# ---------------------------------------------------------------------------
# 1) 바닥 마커 재보정
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not C.aruco_available(), reason="이 OpenCV에 ArUco가 없음")
@pytest.mark.parametrize("cam_i,layout", [(0, FRONT_MARKERS), (1, REAR_MARKERS)])
@pytest.mark.parametrize("err", [(0.0, 0.0, 0.0), (1.5, 2.0, -1.0), (-3.0, -1.0, 2.0)])
def test_markers_find_tilt(cam_i, layout, err):
    nom = load_config(CUR)[0][cam_i]
    norm, XY, ids = _detect(nom, _marker_frame(_tilt(nom, *err), layout), layout, "aruco")
    assert len(ids) >= 4
    for solve_pos in (False, True):
        pose = C.solve_pose_markers(nom, norm, XY, solve_position=solve_pos)
        d = C.pose_delta(nom, pose)
        assert d["yaw"] == pytest.approx(err[0], abs=0.2)
        assert d["pitch"] == pytest.approx(err[1], abs=0.2)
        assert d["roll"] == pytest.approx(err[2], abs=0.2)
        assert d["position_m"] < 0.05
    # 지면 역투영 오차: 틀어진 만큼 컸다가(틀었을 때) 새 값에서 1cm 아래로
    new = C._cam_with(nom, pose["yaw"], pose["pitch"], pose["roll"], pose["position"])
    assert C.marker_errors(new, norm, XY)["ground_m_mean"] < 0.01
    if err != (0.0, 0.0, 0.0):
        assert C.marker_errors(nom, norm, XY)["ground_m_mean"] > 0.1


def test_circle_markers_fallback():
    """ArUco가 없을 때 쓰는 원 마커. 원 무게중심의 원근 치우침이 있어 허용 오차를 0.5도로 둔다."""
    nom = load_config(CUR)[0][0]
    err = (1.5, 2.0, -1.0)
    norm, XY, ids = _detect(nom, _marker_frame(_tilt(nom, *err), FRONT_MARKERS, "circle"), FRONT_MARKERS, "circle")
    assert len(ids) >= 5
    d = C.pose_delta(nom, C.solve_pose_markers(nom, norm, XY))
    for k, e in zip(("yaw", "pitch", "roll"), err):
        assert d[k] == pytest.approx(e, abs=0.5)


@pytest.mark.skipif(not C.aruco_available(), reason="이 OpenCV에 ArUco가 없음")
def test_recalib_script_write_only_when_asked(tmp_path):
    """--write 없이는 설정을 그대로 두고, 주면 각도 줄만 바꾸며 주석과 다른 줄은 남긴다. 큰 변화는 거절한다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("recalib_markers", os.path.join(ROOT, "scripts", "recalib_markers.py"))
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)

    cfg = tmp_path / "cfg.yaml"
    shutil.copy(CUR, cfg)
    orig = cfg.read_text(encoding="utf-8")
    nom = load_config(str(cfg))[0][0]
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), _marker_frame(_tilt(nom, 1.5, 2.0, -1.0), FRONT_MARKERS))
    lay = tmp_path / "lay.csv"
    lay.write_text("id,X,Y\n" + "".join(f"{i},{X},{Y}\n" for i, X, Y in FRONT_MARKERS))
    base = ["--config", str(cfg), "--camera", "front", "--image", str(img), "--layout", str(lay),
            "--backup-dir", str(tmp_path / "bak"), "--json", str(tmp_path / "r.json")]
    assert rm.main(base) == 0
    assert cfg.read_text(encoding="utf-8") == orig  # 확인만: 그대로
    assert rm.main(base + ["--write", "--max-change-deg", "1.0"]) == 1  # 2도 바뀜 > 1도: 거절
    assert cfg.read_text(encoding="utf-8") == orig
    assert rm.main(base + ["--write"]) == 0
    new_text = cfg.read_text(encoding="utf-8")
    cams = load_config(str(cfg))[0]
    assert cams[0].pitch_deg == pytest.approx(3.0 + 2.0, abs=0.2)
    assert cams[0].yaw_deg == pytest.approx(1.5, abs=0.2)
    assert cams[0].roll_deg == pytest.approx(-1.0, abs=0.2)
    assert cams[1].pitch_deg == 20.0 and cams[1].yaw_deg == 180.0  # 다른 카메라는 그대로
    assert np.allclose(cams[0].position, nom.position)  # 각도만 풀었으니 위치 그대로
    # 바뀐 줄은 각도 세 줄뿐이고 주석 수도 같다
    diff = [(a, b) for a, b in zip(orig.splitlines(), new_text.splitlines()) if a != b]
    assert len(diff) == 3 and all("#" in b for a, b in diff if "#" in a)
    assert orig.count("#") == new_text.count("#")
    assert len(os.listdir(tmp_path / "bak")) == 1  # 원래 파일 사본


def test_update_extrinsics_position_car():
    """모형 설정(position_car, 실차 m)은 위치를 축척으로 나눠 적는다."""
    path = os.path.join(ROOT, "configs", "two_cam_model.yaml")
    text = open(path, encoding="utf-8").read()
    s = yaml.safe_load(text)["scene"]["scale"]
    pose = {"position": np.array([0.18, -0.006, 0.150]), "yaw": 0.5, "pitch": 4.0, "roll": 0.2}
    new = C.update_extrinsics_text(text, "front", pose, True, s)
    cam = yaml.safe_load(new)["cameras"][0]["extrinsics"]
    assert cam["position_car"] == pytest.approx([0.18 / s, -0.006 / s, 0.150 / s], abs=1e-3)
    assert (cam["yaw"], cam["pitch"], cam["roll"]) == (0.5, 4.0, 0.2)
    assert yaml.safe_load(new)["cameras"][1] == yaml.safe_load(text)["cameras"][1]


# ---------------------------------------------------------------------------
# 2) 주행 중 보정 점검(차선 소실점)
# ---------------------------------------------------------------------------
def _run_monitor(true_cams, cams, grid, frames=6):
    ipm = IpmCompositor(cams, grid)
    mon = CalibMonitor(cams, grid, ipm.owner, cfg=CalibMonitorConfig(min_samples=2))
    world = SimWorld(true_cams, objects=[])
    warns = []
    for k in range(frames):
        world.set_time(0.37 * k)
        warns = mon.update(ipm.warp(world.render()), 0.37 * k)
    return mon.status(), warns


@pytest.mark.parametrize("err", [(0.0, 0.0), (1.5, 2.0), (-1.0, -2.5), (2.5, -1.0)])
def test_monitor_front(err):
    cams, grid, _ = load_config(CUR)
    st, warns = _run_monitor([_tilt(cams[0], *err), cams[1]], cams, grid)
    f = st[0]
    assert f["n"] >= 2
    assert f["yaw"] == pytest.approx(err[0], abs=0.4)
    assert f["pitch"] == pytest.approx(err[1], abs=0.4)
    assert f["warn"] == (err != (0.0, 0.0))
    assert any("front" in w for w in warns) == (err != (0.0, 0.0))
    if err != (0.0, 0.0):
        # 아래로 더 숙이면(pitch +) 소실점(지평선)이 영상 위로 올라간다
        assert np.sign(f["horizon_px"]) == -np.sign(err[1])


def test_monitor_rear():
    """후방(화각 좁고 많이 숙임)은 차선이 덜 잡히지만 잡히면 같은 방법으로 잰다."""
    cams, grid, _ = load_config(CUR)
    st, _ = _run_monitor([cams[0], _tilt(cams[1], 1.0, -2.0)], cams, grid, frames=8)
    r = st[1]
    assert r["n"] >= 2
    assert r["yaw"] == pytest.approx(1.0, abs=0.4)
    assert r["pitch"] == pytest.approx(-2.0, abs=0.4)
    assert r["warn"] and not st[0]["warn"]


def test_pipeline_calib_monitor_option():
    """파이프라인 옵션: 기본은 꺼짐(None), 켜면 틀어진 전방 카메라를 경고한다."""
    cams, grid, _ = load_config(CUR)
    world = SimWorld([_tilt(cams[0], 0.0, 2.0), cams[1]], objects=[])
    off = RealityViewPipeline(cams, SimBackend(world), grid)
    world.render()
    assert off.step(world.render(), 0.1).calib_warnings is None
    pipe = RealityViewPipeline(cams, SimBackend(world), grid,
                               calib_monitor=CalibMonitorConfig(every=1, min_samples=2))
    out = None
    for _ in range(4):
        world.step(0.37)
        out = pipe.step(world.render(), 0.37)
    assert out.calib_warnings and "front" in out.calib_warnings[0]
    assert pipe.calib_monitor.summary()["used"]["front"] >= 2
    # 화면 왼쪽 아래에 경고 글이 실제로 그려지는지(run_demo가 대시보드에 하는 것과 같은 호출)
    from realityview.calib_monitor import draw_warnings

    dash = out.dashboard.copy()
    draw_warnings(dash, out.calib_warnings)
    h = dash.shape[0]
    assert np.any(dash[h - 40:, :300] != out.dashboard[h - 40:, :300])


# ---------------------------------------------------------------------------
# 3) 카메라 간 겹침 확인
# ---------------------------------------------------------------------------
def test_overlap_pairs():
    cams, grid, _ = load_config(CUR)
    assert overlap_pairs(cams, grid) == []  # 전방, 후방 사이는 사각지대
    cams4, grid4, _ = load_config(FOUR)
    names = [(cams4[i].name, cams4[j].name) for i, j, _ in overlap_pairs(cams4, grid4)]
    assert ("rear", "left") in names and ("rear", "right") in names


@pytest.mark.parametrize("err", [(0.0, 0.0), (2.0, 0.0), (-3.0, 1.0), (0.0, 2.0)])
def test_overlap_score_finds_tilt(err):
    cams, grid, _ = load_config(FOUR)
    names = [c.name for c in cams]
    L, R = names.index("left"), names.index("rear")
    true = list(cams)
    true[L] = _tilt(cams[L], *err)
    world = SimWorld(true, objects=[])
    world.set_time(0.3)
    s = overlap_score(cams, world.render(), grid, R, L)
    assert s["cells"] > 1000
    if err == (0.0, 0.0):
        assert s["ok"] is True and s["shift_m"] < 0.15 and s["zncc"] > 0.9
    else:
        assert s["ok"] is False
        assert s["zncc"] < 0.8
        assert s["shift_m"] is None or s["shift_m"] > 0.25
