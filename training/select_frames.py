#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    RoundPaths,
    check_round_name,
    load_config,
    make_frame_id,
    read_csv,
    read_json,
    resolve_path,
    sanitize,
    segment_key,
    update_meta,
    write_csv,
    write_json,
)

# 주행 녹화와 트리거 클립에서 학습용 프레임을 고르고, 고정 시험 세트를 정해 잠근다.
# numpy 와 opencv 만 쓰므로 맥에서도 돈다.
#
# 예) 저장소 폴더에서
#   python3 -m training.select_frames --config training/configs/round.yaml
#   python3 -m training.select_frames --config training/configs/round.yaml --make-test-set   # 시험 세트만
#
# 결과: out/training/datasets/<회차>/images/<프레임 이름>.jpg, frames.csv, meta.json
#       out/training/test_set/images/, test_set.csv, lock.json (처음 한 번)

FRAME_FIELDS = [
    "frame_id", "drive", "segment", "camera", "idx", "t", "source", "trigger_kind", "trigger_score",
    "blur", "brightness", "scene", "scene_by", "seg_key",
]
TEST_FIELDS = ["frame_id", "drive", "segment", "camera", "idx", "t", "scene", "seg_key", "added_round", "label_source"]


# ---------------------------------------------------------------------------
# 입력 찾기: 주행 녹화와 트리거 클립을 "스트림"(카메라 하나의 연속 프레임)으로 정리
# ---------------------------------------------------------------------------
@dataclass
class Stream:
    kind: str  # "video": path 가 영상 파일, "images": images 가 (번호, 경로) 목록,
    #            "mjpeg": path 가 JPEG 를 이어 붙인 파일, images 가 (번호, 바이트 위치, 크기) 목록
    drive: str
    segment: str
    camera: str
    source: str = "drive"  # "drive" 또는 "trigger"
    path: str | None = None
    images: list = field(default_factory=list)
    times: dict = field(default_factory=dict)  # 프레임 번호 -> 세그먼트 시작부터의 초
    fps: float = 0.0
    trigger_kind: str = ""
    trigger_score: float | None = None


def _load_times(seg_dir: str, camera: str, rec: dict) -> dict:
    """세그먼트 폴더에서 이 카메라의 타임스탬프 csv 를 찾아 {프레임 번호: 초} 로 읽는다.

    열 이름은 설정(recording.frame_columns, time_columns)에서 처음 맞는 것을 쓴다. 열 이름이 ns 로
    끝나면 나노초로 보고 초로 바꾼다. 첫 프레임을 0초로 맞춘다. csv 가 없으면 빈 dict(프레임 번호 / fps 로 대신함).
    """
    for pat in rec["timestamp_csv"]:
        p = os.path.join(seg_dir, pat.format(camera=camera))
        rows = read_csv(p)
        if not rows:
            continue
        cols = rows[0].keys()
        fcol = next((c for c in rec["frame_columns"] if c in cols), None)
        tcol = next((c for c in rec["time_columns"] if c in cols), None)
        ccol = rec.get("camera_column")
        if tcol is None:
            continue
        out = {}
        for i, r in enumerate(rows):
            # 여러 카메라가 한 csv 에 있으면 카메라 열로 거른다
            if ccol and ccol in cols and r.get(ccol) not in (None, "", camera):
                continue
            try:
                idx = int(float(r[fcol])) if fcol else len(out)
                out[idx] = float(r[tcol])
            except (TypeError, ValueError):
                continue
        if not out:
            continue
        # 단위는 열 이름으로 정한다(sensor_ns 처럼 ns 로 끝나면 나노초). 파이의 sensor_ns 는 부팅 뒤
        # monotonic 시계라 부팅 직후에는 값이 작아서, 크기로 단위를 추측하면 틀린다.
        t0 = min(out.values())
        scale = 1e-9 if tcol.lower().endswith("ns") else 1.0
        return {k: (v - t0) * scale for k, v in out.items()}
    return {}


def _fps_from_times(times: dict) -> float:
    if len(times) < 2:
        return 0.0
    ks = sorted(times)
    dt = (times[ks[-1]] - times[ks[0]]) / max(ks[-1] - ks[0], 1)
    fps = 1.0 / dt if dt > 0 else 0.0
    return fps if 0.5 < fps < 1000 else 0.0  # 말이 안 되는 값이면 버리고 다른 방법(영상 정보, 기본값)으로


def _load_mjpeg_refs(seg_dir: str, camera: str, mjpeg_path: str, rec: dict) -> list:
    """파이 녹화 형식(<카메라>.mjpeg + <카메라>.csv 의 offset, size 열)의 프레임 위치 목록.

    전원이 꺼져 파일 끝이 잘린 경우를 대비해 파일 크기를 넘는 줄은 버린다. 형식이 아니면 빈 목록.
    """
    size_total = os.path.getsize(mjpeg_path)
    for pat in rec["timestamp_csv"]:
        rows = read_csv(os.path.join(seg_dir, pat.format(camera=camera)))
        if not rows or "offset" not in rows[0] or "size" not in rows[0]:
            continue
        fcol = next((c for c in rec["frame_columns"] if c in rows[0]), None)
        refs = []
        for i, r in enumerate(rows):
            try:
                off, sz = int(r["offset"]), int(r["size"])
                idx = int(float(r[fcol])) if fcol else i
            except (TypeError, ValueError):
                continue
            if sz > 0 and off + sz <= size_total:
                refs.append((idx, off, sz))
        return refs
    return []


def streams_in_dir(seg_dir: str, drive: str, segment: str, rec: dict, source: str = "drive") -> list[Stream]:
    """폴더 하나에서 카메라별 스트림을 만든다.

    영상 파일(<카메라>.mp4 등)은 파일 하나가 카메라 하나다. 이미지 파일은 record_frames.py 처럼
    <카메라>_<번호>.png 이름을 카메라별로 묶는다(설정 recording.image_pattern).
    """
    out: list[Stream] = []
    if not os.path.isdir(seg_dir):
        return out
    names = sorted(os.listdir(seg_dir))
    exts = tuple(e.lower() for e in rec["video_exts"])
    img_re = re.compile(rec["image_pattern"])
    groups: dict = {}
    for n in names:
        p = os.path.join(seg_dir, n)
        stem, ext = os.path.splitext(n)
        if not os.path.isfile(p):
            continue
        if ext.lower() in exts:
            times = _load_times(seg_dir, stem, rec)
            refs = _load_mjpeg_refs(seg_dir, stem, p, rec) if ext.lower() in (".mjpeg", ".mjpg") else []
            if refs:
                out.append(Stream("mjpeg", drive, segment, stem, source, path=p, images=refs, times=times))
            else:
                out.append(Stream("video", drive, segment, stem, source, path=p, times=times))
        elif ext.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            m = img_re.match(stem)
            cam, idx = (m.group("camera"), int(m.group("idx"))) if m else ("cam", len(groups.get("cam", [])))
            groups.setdefault(cam, []).append((idx, p))
    for cam, items in sorted(groups.items()):
        times = _load_times(seg_dir, cam, rec)
        out.append(Stream("images", drive, segment, cam, source, images=sorted(items), times=times))
    return out


def discover_drive_streams(cfg: dict) -> list[Stream]:
    """drives_root/<주행>/<세그먼트>/ 에서 설정 drives 에 맞는 주행의 스트림을 모두 찾는다.

    주행 폴더 바로 아래에 영상이 있으면(세그먼트 없이 녹화한 경우) 세그먼트 이름을 "seg0" 으로 둔다.
    """
    root = RoundPaths(cfg).drives_root
    rec = cfg["recording"]
    pats = cfg.get("drives") or ["*"]
    out: list[Stream] = []
    if not os.path.isdir(root):
        return out
    for drive in sorted(os.listdir(root)):
        ddir = os.path.join(root, drive)
        if not os.path.isdir(ddir) or not any(fnmatch.fnmatch(drive, p) for p in pats):
            continue
        out.extend(streams_in_dir(ddir, drive, "seg0", rec))
        for seg in sorted(os.listdir(ddir)):
            sdir = os.path.join(ddir, seg)
            if os.path.isdir(sdir):
                out.extend(streams_in_dir(sdir, drive, seg, rec))
    return out


def _first_key(d: dict, keys) -> object:
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def discover_trigger_streams(cfg: dict) -> list[Stream]:
    """triggers_root 아래 json 마다 클립 스트림을 만든다.

    json 형식은 파이 쪽 12번 작업이 정하므로(확인 필요) 키 이름을 설정에서 받는다.
      - 종류, 점수: triggers.kind_keys, score_keys 에서 처음 있는 키
      - 영상이나 이미지 폴더: triggers.media_keys 에서 처음 있는 키(json 기준 상대 경로)
      - 그런 키가 없으면 json 과 같은 폴더의 영상과 이미지를 클립으로 본다
    """
    tcfg = cfg["triggers"]
    root = RoundPaths(cfg).triggers_root
    out: list[Stream] = []
    if not tcfg.get("enabled", True) or not os.path.isdir(root):
        return out
    rec = cfg["recording"]
    for jp in sorted(glob.glob(os.path.join(root, tcfg["json_glob"]), recursive=True)):
        meta = read_json(jp, {}) or {}
        jdir = os.path.dirname(jp)
        stem = os.path.splitext(os.path.basename(jp))[0]
        clip = stem if stem.lower() not in ("meta", "trigger", "info", "clip", "event") else os.path.basename(jdir)
        # 파이 트리거 클립(realityview/triggers.py)은 종류와 점수를 events 목록 안에 둔다.
        # 위에 없으면 events 의 종류를 모아 "+" 로 잇고, 점수는 가장 낮은 값을 쓴다.
        events = [e for e in (meta.get(tcfg.get("events_key", "events")) or []) if isinstance(e, dict)]
        kind = _first_key(meta, tcfg["kind_keys"])
        if kind is None and events:
            kind = "+".join(sorted({str(_first_key(e, tcfg["kind_keys"])) for e in events}))
        kind = str(kind or "unknown")
        score = _first_key(meta, tcfg["score_keys"])
        if score is None and events:
            vals = [_first_key(e, tcfg["score_keys"]) for e in events]
            vals = [v for v in vals if isinstance(v, (int, float))]
            score = min(vals) if vals else None
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        media = _first_key(meta, tcfg["media_keys"])
        streams: list[Stream] = []
        if isinstance(media, str):
            mp = media if os.path.isabs(media) else os.path.join(jdir, media)
            if os.path.isdir(mp):
                streams = streams_in_dir(mp, "trig", clip, rec, source="trigger")
            elif os.path.isfile(mp):
                cam = str(_first_key(meta, tcfg["camera_keys"]) or os.path.splitext(os.path.basename(mp))[0])
                streams = [Stream("video", "trig", clip, cam, "trigger", path=mp)]
        if not streams:
            streams = streams_in_dir(jdir, "trig", clip, rec, source="trigger")
        for s in streams:
            s.trigger_kind, s.trigger_score = kind, score
        out.extend(streams)
    return out


# ---------------------------------------------------------------------------
# 프레임 읽기: 앞에서부터 차례로 읽고 N번째마다 하나씩 (탐색하지 않아 csv 번호와 어긋나지 않음)
# ---------------------------------------------------------------------------
def stream_fps(s: Stream, cap=None, fallback: float = 30.0) -> float:
    if s.fps > 0:
        return s.fps
    fps = 0.0
    if cap is not None:
        v = cap.get(cv2.CAP_PROP_FPS)
        fps = v if 0.5 < v < 1000 else 0.0
    if fps <= 0:
        fps = _fps_from_times(s.times)
    s.fps = fps if fps > 0 else fallback
    return s.fps


def iter_frames(s: Stream, every_sec: float, fallback_fps: float, wanted: set | None = None):
    """(프레임 번호, 초, BGR 영상)을 내놓는다. wanted 가 있으면 그 번호만, 없으면 every_sec 간격으로."""
    if s.kind == "video":
        cap = cv2.VideoCapture(s.path)
        if not cap.isOpened():
            print(f"[경고] 영상을 열지 못했습니다: {s.path}", file=sys.stderr)
            return
        fps = stream_fps(s, cap, fallback_fps)
        step = max(1, int(round(every_sec * fps)))
        idx = 0
        try:
            while True:
                if wanted is not None and idx > max(wanted, default=-1):
                    break
                ok = cap.grab()
                if not ok:
                    break
                take = (idx in wanted) if wanted is not None else (idx % step == 0)
                if take:
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        yield idx, s.times.get(idx, idx / fps), img
                idx += 1
        finally:
            cap.release()
    elif s.kind == "mjpeg":
        # csv 의 바이트 위치로 JPEG 한 장씩 바로 꺼낸다(앞 프레임을 풀 필요 없음)
        fps = stream_fps(s, None, fallback_fps)
        step = max(1, int(round(every_sec * fps)))
        with open(s.path, "rb") as fh:
            for n, (idx, off, size) in enumerate(s.images):
                take = (idx in wanted) if wanted is not None else (n % step == 0)
                if not take:
                    continue
                fh.seek(off)
                img = cv2.imdecode(np.frombuffer(fh.read(size), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield idx, s.times.get(idx, idx / fps), img
    else:
        fps = stream_fps(s, None, fallback_fps)
        step = max(1, int(round(every_sec * fps)))
        for n, (idx, p) in enumerate(s.images):
            take = (idx in wanted) if wanted is not None else (n % step == 0)
            if not take:
                continue
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                yield idx, s.times.get(idx, idx / fps), img


# ---------------------------------------------------------------------------
# 프레임 성질: 흐림, 밝기, 축소 영상
# ---------------------------------------------------------------------------
def frame_features(img: np.ndarray, blur_width: int = 640) -> tuple[float, float, np.ndarray]:
    """(라플라시안 분산, 평균 밝기, 32x18 축소 회색 영상).

    라플라시안 분산은 경계가 또렷할수록 크다. 해상도마다 값이 달라지지 않게 폭 blur_width 로 줄여서 잰다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.shape[1] > blur_width:
        h = int(round(gray.shape[0] * blur_width / gray.shape[1]))
        gray = cv2.resize(gray, (blur_width, h), interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return blur, bright, thumb


def load_scene_tags(path: str | None) -> list[dict]:
    """수동 태그 csv: drive, segment, (camera), start_s, end_s, scene. 빈 칸 camera 는 모든 카메라."""
    rows = read_csv(resolve_path(path)) if path else []
    out = []
    for r in rows:
        try:
            out.append({
                "drive": sanitize(r.get("drive", "")), "segment": sanitize(r.get("segment", "")),
                "camera": sanitize(r["camera"]) if r.get("camera") else "",
                "start": float(r.get("start_s") or 0), "end": float(r.get("end_s") or 1e18),
                "scene": (r.get("scene") or "").strip(),
            })
        except ValueError:
            continue
    return [r for r in out if r["scene"]]


def scene_of(tags: list[dict], drive: str, segment: str, camera: str, t: float, bright: float, night: float):
    """(장면 이름, 정한 방법). 수동 태그가 맞으면 태그, 아니면 밝기로 day/night."""
    d, s, c = sanitize(drive), sanitize(segment), sanitize(camera)
    for r in tags:
        if r["drive"] == d and r["segment"] in (s, "") and r["camera"] in (c, "") and r["start"] <= t <= r["end"]:
            return r["scene"], "tag"
    return ("night" if bright < night else "day"), "brightness"


# ---------------------------------------------------------------------------
# 후보 만들기: 간격 뽑기, 흐린 프레임 빼기, 거의 같은 프레임 빼기, 장면 정하기
# ---------------------------------------------------------------------------
def scan_stream(s: Stream, cfg: dict, tags: list[dict], stats: dict) -> list[dict]:
    sel = cfg["select"]
    every = cfg["triggers"]["every_sec"] if s.source == "trigger" else sel["every_sec"]
    kept_thumbs: list[np.ndarray] = []
    out = []
    for idx, t, img in iter_frames(s, every, cfg["recording"]["fps_fallback"]):
        stats["seen"] = stats.get("seen", 0) + 1
        blur, bright, thumb = frame_features(img, sel["blur_width"])
        # 어두운 영상은 경계 대비가 작아 라플라시안 분산도 작으므로 밤 문턱을 따로 둔다
        blur_min = sel["blur_min_night"] if bright < sel["night_brightness"] else sel["blur_min"]
        if blur < blur_min:
            stats["blur_dropped"] = stats.get("blur_dropped", 0) + 1
            continue
        # 같은 카메라에서 최근에 남긴 프레임과 거의 같으면(정차 중 등) 버린다
        if kept_thumbs and min(float(np.abs(thumb - k).mean()) for k in kept_thumbs) < sel["dup_diff"]:
            stats["dup_dropped"] = stats.get("dup_dropped", 0) + 1
            continue
        kept_thumbs.append(thumb)
        kept_thumbs = kept_thumbs[-int(sel["dup_window"]):]
        scene, by = scene_of(tags, s.drive, s.segment, s.camera, t, bright, sel["night_brightness"])
        out.append({
            "frame_id": make_frame_id(s.drive, s.segment, s.camera, idx),
            "drive": s.drive, "segment": s.segment, "camera": s.camera, "idx": idx, "t": round(float(t), 4),
            "source": s.source, "trigger_kind": s.trigger_kind,
            "trigger_score": "" if s.trigger_score is None else s.trigger_score,
            "blur": round(blur, 2), "brightness": round(bright, 2), "scene": scene, "scene_by": by,
            "seg_key": segment_key(s.drive, s.segment),
            "_stream": s,
        })
    return out


def scan_all(streams: list[Stream], cfg: dict) -> tuple[list[dict], dict]:
    tags = load_scene_tags(cfg["select"].get("scene_tags_csv"))
    stats: dict = {}
    cands: list[dict] = []
    for s in streams:
        cands.extend(scan_stream(s, cfg, tags, stats))
    stats["candidates"] = len(cands)
    return cands, stats


# ---------------------------------------------------------------------------
# 고르게 뽑기
# ---------------------------------------------------------------------------
def balanced_pick(cands: list[dict], n: int, key: str = "scene") -> list[dict]:
    """key(장면 등)마다 같은 몫을 주고, 모자란 묶음의 남는 몫은 다른 묶음에 넘긴다.

    묶음 안에서는 (세그먼트, 카메라, 시각) 순으로 늘어놓고 일정 간격으로 골라 시간상 고르게 퍼지게 한다.
    난수를 쓰지 않으므로 같은 입력이면 항상 같은 결과다.
    """
    if n <= 0 or not cands:
        return []
    groups: dict = {}
    for c in cands:
        groups.setdefault(str(c.get(key, "")), []).append(c)
    quota = {g: 0 for g in groups}
    left = min(n, len(cands))
    # 물 채우기: 아직 여유 있는 묶음에 한 장씩 돌아가며 준다
    while left > 0:
        open_groups = [g for g in sorted(groups) if quota[g] < len(groups[g])]
        if not open_groups:
            break
        share = max(1, left // len(open_groups))
        for g in open_groups:
            add = min(share, len(groups[g]) - quota[g], left)
            quota[g] += add
            left -= add
            if left <= 0:
                break
    out = []
    for g in sorted(groups):
        items = sorted(groups[g], key=lambda c: (c["seg_key"], c["camera"], float(c["t"])))
        k = quota[g]
        if k <= 0:
            continue
        pos = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
        chosen = [items[i] for i in pos]
        # 반올림으로 겹쳐 모자라면 앞에서부터 채운다
        rest = [c for i, c in enumerate(items) if i not in set(pos.tolist())]
        chosen += rest[: k - len(chosen)]
        out.extend(chosen)
    return out


# ---------------------------------------------------------------------------
# 고정 시험 세트: 정하기, 잠그기, 제외 판정
# ---------------------------------------------------------------------------
class HoldoutLock:
    """lock.json 을 읽어 어떤 프레임이 학습에서 빠져야 하는지 판정한다.

    exclude=segment: 시험 프레임이 나온 세그먼트(모든 카메라)를 통째로 뺀다. 1초 옆 프레임은 거의 같은
                     장면이라 프레임만 빼면 시험 점수가 부풀기 때문이다.
    exclude=window:  같은 세그먼트에서 시험 프레임 앞뒤 window_sec 초를 뺀다.
    """

    def __init__(self, data: dict | None):
        self.data = data or {}
        self.mode = self.data.get("exclude", "segment")
        self.window = float(self.data.get("window_sec", 10.0))
        self.segments = set(self.data.get("segments", []))
        self.frames = {f["frame_id"]: f for f in self.data.get("frames", [])}
        self.by_seg: dict = {}
        for f in self.frames.values():
            self.by_seg.setdefault(f["seg_key"], []).append(float(f["t"]))

    @property
    def exists(self) -> bool:
        return bool(self.frames)

    def excludes(self, frame_id: str, seg_key: str | None, t: float | None) -> bool:
        if frame_id in self.frames:
            return True
        if seg_key is None:
            return False
        if self.mode == "segment":
            return seg_key in self.segments
        if t is None:
            return seg_key in self.by_seg
        return any(abs(float(t) - x) <= self.window for x in self.by_seg.get(seg_key, []))


def load_lock(paths: RoundPaths) -> HoldoutLock:
    data = read_json(paths.test_lock)
    lock = HoldoutLock(data)
    if data:
        # 잠근 뒤 누가 목록을 바꿨는지 확인(라벨 출처 열은 바뀌어도 되므로 프레임 이름만 비교)
        ids = sorted(r["frame_id"] for r in read_csv(paths.test_csv))
        if ids != sorted(lock.frames):
            raise RuntimeError(
                f"시험 세트 목록({paths.test_csv})이 잠금 파일과 다릅니다. 손으로 고쳤다면 되돌리세요."
            )
    return lock


def choose_test_segments(cands: list[dict], ratio: float) -> list[str]:
    """시험용으로 떼어 둘 세그먼트를 장면별로 돌아가며 고른다. 세그먼트가 2개 이상이면 하나는 학습에 남긴다."""
    seg_scene: dict = {}
    for c in cands:
        seg_scene.setdefault(c["seg_key"], []).append(c["scene"])
    segs = sorted(seg_scene)
    if not segs:
        return []
    k = max(1, int(round(ratio * len(segs))))
    if len(segs) >= 2:
        k = min(k, len(segs) - 1)
    by_scene: dict = {}
    for s in segs:
        vals, counts = np.unique(seg_scene[s], return_counts=True)
        by_scene.setdefault(str(vals[int(np.argmax(counts))]), []).append(s)
    # 장면마다 세그먼트를 시간 순으로 두고, 가운데부터 고른다(앞뒤 세그먼트와 덜 닮게)
    queues = {g: sorted(v, key=lambda x: (abs(v.index(x) - len(v) // 2), x)) for g, v in by_scene.items()}
    chosen: list[str] = []
    while len(chosen) < k:
        moved = False
        for g in sorted(queues):
            if queues[g] and len(chosen) < k:
                chosen.append(queues[g].pop(0))
                moved = True
        if not moved:
            break
    return sorted(chosen)


def make_test_set(cfg: dict, cands: list[dict], append: bool = False) -> dict:
    """후보에서 시험 세트를 고르고 이미지와 목록을 쓰고 잠근다. 이미 잠겼으면 append 일 때만 덧붙인다."""
    paths = RoundPaths(cfg)
    tcfg = cfg["test_set"]
    old = load_lock(paths)
    if old.exists and not append:
        raise RuntimeError(f"시험 세트가 이미 잠겨 있습니다({paths.test_lock}). 덧붙이려면 --add-to-test-set")
    pool = [c for c in cands if tcfg.get("include_triggers") or c["source"] == "drive"]
    pool = [c for c in pool if not old.excludes(c["frame_id"], c["seg_key"], c["t"])]
    # 이미 시험 세트를 낸 세그먼트에서 또 고르지 않는다(덧붙일 때는 새 주행에서만)
    pool = [c for c in pool if c["seg_key"] not in old.by_seg]
    mode = old.mode if old.exists else tcfg["exclude"]
    if mode == "segment":
        segs = choose_test_segments(pool, float(tcfg["segment_ratio"]))
        pool = [c for c in pool if c["seg_key"] in set(segs)]
    else:
        segs = []
    picked = balanced_pick(pool, int(tcfg["n_frames"]), key="scene")
    save_images(picked, os.path.join(paths.test_dir, "images"), cfg["select"]["jpeg_quality"], cfg)
    rows = read_csv(paths.test_csv)
    for c in picked:
        rows.append({**{k: c[k] for k in TEST_FIELDS if k in c}, "added_round": cfg["round"], "label_source": "none"})
    write_csv(paths.test_csv, rows, TEST_FIELDS)
    frames = list(old.data.get("frames", [])) + [
        {"frame_id": c["frame_id"], "seg_key": c["seg_key"], "t": c["t"], "scene": c["scene"],
         "added_round": cfg["round"]} for c in picked
    ]
    lock = {
        "exclude": mode,
        "window_sec": float(old.data.get("window_sec", tcfg["window_sec"])),
        "segments": sorted(set(old.data.get("segments", [])) | set(segs)),
        "frames": frames,
        "rounds": sorted(set(old.data.get("rounds", [])) | {cfg["round"]}),
    }
    write_json(paths.test_lock, lock)
    return {"added": len(picked), "total": len(frames), "segments": lock["segments"], "mode": mode}


# ---------------------------------------------------------------------------
# 이미지 저장 (두 번째로 읽으며 고른 번호만 저장)
# ---------------------------------------------------------------------------
def save_images(picked: list[dict], out_dir: str, quality: int, cfg: dict) -> int:
    os.makedirs(out_dir, exist_ok=True)
    by_stream: dict = {}
    for c in picked:
        by_stream.setdefault(id(c["_stream"]), (c["_stream"], {}))[1][int(c["idx"])] = c["frame_id"]
    n = 0
    for s, want in by_stream.values():
        for idx, _t, img in iter_frames(s, 1.0, cfg["recording"]["fps_fallback"], wanted=set(want)):
            cv2.imwrite(os.path.join(out_dir, want[idx] + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            n += 1
    return n


# ---------------------------------------------------------------------------
# 한 회차 선별
# ---------------------------------------------------------------------------
def used_in_previous_rounds(paths: RoundPaths) -> set:
    """이전 회차 frames.csv 에 있는 프레임 이름(같은 주행을 다시 적어도 두 번 쓰지 않게)."""
    me = check_round_name(paths.round)
    used = set()
    if not os.path.isdir(paths.datasets):
        return used
    for name in os.listdir(paths.datasets):
        try:
            if check_round_name(name) >= me:
                continue
        except ValueError:
            continue
        used.update(r["frame_id"] for r in read_csv(os.path.join(paths.datasets, name, "frames.csv")))
    return used


def select_round(cfg: dict, make_test: bool = True) -> dict:
    """시험 세트가 없으면 먼저 정해 잠그고, 그 세그먼트를 뺀 나머지에서 이번 회차 프레임을 고른다."""
    paths = RoundPaths(cfg)
    sel = cfg["select"]
    streams = discover_drive_streams(cfg) + discover_trigger_streams(cfg)
    if not streams:
        raise RuntimeError(f"녹화를 찾지 못했습니다: {paths.drives_root}, {paths.triggers_root}")
    cands, stats = scan_all(streams, cfg)

    lock = load_lock(paths)
    test_info = None
    if not lock.exists:
        if not make_test:
            raise RuntimeError("고정 시험 세트가 아직 없습니다. 먼저 --make-test-set 으로 정하세요.")
        test_info = make_test_set(cfg, cands)
        lock = load_lock(paths)
        print(f"고정 시험 세트를 정하고 잠갔습니다: {test_info['added']}장, 세그먼트 {test_info['segments']}")

    used = used_in_previous_rounds(paths)
    pool = []
    for c in cands:
        if lock.excludes(c["frame_id"], c["seg_key"] if c["source"] == "drive" else None, c["t"]):
            stats["test_excluded"] = stats.get("test_excluded", 0) + 1
        elif c["frame_id"] in used:
            stats["prev_round_excluded"] = stats.get("prev_round_excluded", 0) + 1
        else:
            pool.append(c)

    n = int(sel["n_frames"])
    trig = [c for c in pool if c["source"] == "trigger"]
    normal = [c for c in pool if c["source"] != "trigger"]
    # 트리거 프레임 먼저(모델이 약한 장면), 종류별로 고르게. 남은 몫은 장면별로 고르게.
    picked = balanced_pick(trig, min(len(trig), int(n * float(sel["trigger_max_ratio"]))), key="trigger_kind")
    picked += balanced_pick(normal, n - len(picked), key="scene")

    save_images(picked, paths.images, sel["jpeg_quality"], cfg)
    write_csv(paths.frames_csv, picked, FRAME_FIELDS)
    scenes: dict = {}
    for c in picked:
        scenes[c["scene"]] = scenes.get(c["scene"], 0) + 1
    info = {
        **stats,
        "selected": len(picked),
        "selected_triggers": sum(1 for c in picked if c["source"] == "trigger"),
        "scenes": scenes,
        "drives": sorted({c["drive"] for c in picked if c["source"] == "drive"}),
        "test_set_created": test_info,
    }
    update_meta(paths, "select", info)
    return info


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="학습 프레임 선별과 고정 시험 세트")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--make-test-set", action="store_true", help="시험 세트만 정하고 잠근 뒤 끝냄")
    ap.add_argument("--add-to-test-set", action="store_true", help="잠긴 시험 세트에 이번 주행에서 덧붙임")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.make_test_set or args.add_to_test_set:
        streams = discover_drive_streams(cfg) + discover_trigger_streams(cfg)
        cands, _ = scan_all(streams, cfg)
        info = make_test_set(cfg, cands, append=args.add_to_test_set)
        print(f"시험 세트: 이번에 {info['added']}장, 전체 {info['total']}장, 방식 {info['mode']}")
        return 0
    info = select_round(cfg)
    print(f"{cfg['round']}: 후보 {info['candidates']}장 중 {info['selected']}장 선택 "
          f"(트리거 {info['selected_triggers']}, 흐림 제외 {info.get('blur_dropped', 0)}, "
          f"중복 제외 {info.get('dup_dropped', 0)}, 시험 세트 제외 {info.get('test_excluded', 0)}), 장면 {info['scenes']}")
    print(f"결과: {RoundPaths(cfg).rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
