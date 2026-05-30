#!/usr/bin/env python3
"""
navigate.py — 基于深度感知的占用率图 + Hybrid A* 重规划避障导航

TCP 通道：
  :12345  Python → UE  SetPose 控制指令（客户端，连接 ATCPServer）
  :8991   UE → Python  位姿遥测（客户端，连接 PoseTelemetryComponent）
  :8989   UE → Python  双目帧流（服务端，UE FrameStreamingComponent 主动连入）
"""

import sys
import os
import json
import math
import socket
import threading
import queue
import time
import warnings
from collections import deque

import copy

import numpy as np
import cv2

# ── 把 scripts 目录加到 sys.path ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from stereo_depth import (
    recv_stereo_frame, decode_jpeg, apply_clahe,
    build_stereo_sgbm, apply_wls_filter,
    disparity_to_depth, disparity_to_float, stereo_confidence_to_float,
    FOCAL_PX, IMAGE_W, IMAGE_H,
    DEFAULT_NUM_DISP, DEFAULT_BLOCK_SIZE, DEFAULT_MIN_DISP,
    DEFAULT_UNIQUENESS, DEFAULT_SPECKLE_W, DEFAULT_SPECKLE_R,
    DEFAULT_WLS_SIGMA, DEFAULT_WLS_LAMBDA,
    HAS_WLS,
)
from planner_hybrid_astar import (
    ConfigurationSpace, CoordinateTransformer,
    HybridAStarPlanner, HybridAStarConfig,
    postprocess_hybrid_path, compute_yaw,
)
from planner_teb_local import TEBLocalPlanner
from occupancy_mapper import OccupancyMapper, DepthFrame, MapStats

# ── 文件路径 ──────────────────────────────────────────────────────
_BASE        = os.path.join(_SCRIPT_DIR, '..')
NAV_CFG_PATH = os.path.join(_BASE, 'config', 'navigation_config.json')

# ── 状态常量 ──────────────────────────────────────────────────────
IDLE, NAV, REACHED_GOAL, FAILED = 'IDLE', 'NAV', 'REACHED_GOAL', 'FAILED'

# ── 录像开关（True=录制 mp4，False=跳过）─────────────────────────
RECORD_VIDEO = False


def normalize_angle_deg(angle: float) -> float:
    """归一化角度到 [-180, 180)。"""
    return (angle + 180.0) % 360.0 - 180.0


def angular_diff_deg(a: float, b: float) -> float:
    return normalize_angle_deg(a - b)


def _fmt_metric(value: float, suffix: str = '', width: int = 4,
                prec: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return '--'
    return f'{value:{width}.{prec}f}{suffix}'


# ══════════════════════════════════════════════════════════════════
# PoseListener：后台线程，连接 UE PoseTelemetryComponent (:8991)
# ══════════════════════════════════════════════════════════════════
class PoseListener(threading.Thread):
    def __init__(self, host: str, port: int, reconnect_s: float):
        super().__init__(daemon=True)
        self._host        = host
        self._port        = port
        self._reconnect_s = reconnect_s
        self._lock        = threading.Lock()
        self._pose        = None   # (x_cm, y_cm, z_cm, nav_yaw_deg)
        self._stop        = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self._host, self._port))
                print(f'[Pose] 已连接 {self._host}:{self._port}')
                buf = b''
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(512)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        self._parse(line.decode('utf-8', errors='ignore').strip())
                sock.close()
            except (ConnectionRefusedError, OSError):
                pass
            if not self._stop.is_set():
                time.sleep(self._reconnect_s)

    def _parse(self, line: str):
        # 格式：Pose:x,y,z,pitch,yaw,roll
        if not line.startswith('Pose:'):
            return
        try:
            vals = [float(v) for v in line[5:].split(',')]
            x, y, z, _pitch, ue_yaw, _roll = vals
            with self._lock:
                self._pose = (x, y, z, ue_yaw)  # actor_yaw == nav_yaw
        except Exception:
            pass

    def get_pose(self):
        with self._lock:
            return self._pose

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════
# FrameWorker：后台线程，监听 :8989，UE 连入后接收双目帧并计算深度
# ══════════════════════════════════════════════════════════════════
class FrameWorker(threading.Thread):
    def __init__(self, listen_host: str, port: int, timeout_s: float,
                 median_window: int = 1):
        super().__init__(daemon=True)
        self._host    = listen_host
        self._port    = port
        self._timeout = timeout_s
        self._q       = queue.Queue(maxsize=1)
        self._stop    = threading.Event()
        self._lock    = threading.Lock()

        self._connected       = False
        self._last_frame_time = None
        self._last_valid_pct  = 0.0
        self._frame_count     = 0
        self._median_window   = max(1, int(median_window))
        self._depth_hist      = deque(maxlen=self._median_window)

        self._stereo = build_stereo_sgbm(
            DEFAULT_NUM_DISP, DEFAULT_BLOCK_SIZE, DEFAULT_MIN_DISP,
            DEFAULT_UNIQUENESS, DEFAULT_SPECKLE_W, DEFAULT_SPECKLE_R)

    def run(self):
        while not self._stop.is_set():
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self._host, self._port))
                srv.listen(1)
                srv.settimeout(2.0)
                print(f'[Frame] 等待 UE 连接 {self._host}:{self._port} ...')
                while not self._stop.is_set():
                    try:
                        conn, addr = srv.accept()
                        break
                    except socket.timeout:
                        continue
                else:
                    srv.close()
                    return
                srv.close()
                with self._lock:
                    self._connected = True
                print(f'[Frame] UE 已连接：{addr}')
                conn.settimeout(self._timeout)
                self._recv_loop(conn)
                conn.close()
                with self._lock:
                    self._connected = False
                if not self._stop.is_set():
                    print('[Frame] 帧流已断开，等待重连...')
            except OSError as e:
                print(f'[Frame] 套接字错误：{e}')
                time.sleep(1.0)

    def _recv_loop(self, conn):
        lmbda_val = DEFAULT_WLS_LAMBDA * 100    # 8000
        sigma_val = DEFAULT_WLS_SIGMA  / 10.0   # 1.5
        use_wls   = HAS_WLS and DEFAULT_WLS_SIGMA > 0

        while not self._stop.is_set():
            try:
                left_jpeg, right_jpeg = recv_stereo_frame(conn)
            except Exception:
                break

            left_bgr  = decode_jpeg(left_jpeg)
            right_bgr = decode_jpeg(right_jpeg)
            if left_bgr is None or right_bgr is None:
                continue

            left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            left_gray  = apply_clahe(left_gray)
            right_gray = apply_clahe(right_gray)

            disp = self._stereo.compute(left_gray, right_gray)
            if use_wls:
                disp, conf_uint8 = apply_wls_filter(
                    self._stereo, left_gray, right_gray, disp,
                    lmbda=lmbda_val, sigma_color=sigma_val)
                _, valid_mask = disparity_to_float(disp)
                confidence = stereo_confidence_to_float(conf_uint8, valid_mask)
            else:
                _, valid_mask = disparity_to_float(disp)
                confidence = valid_mask.astype(np.float32)

            depth = disparity_to_depth(disp)  # float32 H×W，米，0=无效
            depth = self._temporal_filter_depth(depth)
            valid_pct = float(np.count_nonzero(depth > 0) / max(depth.size, 1) * 100.0)
            ts = time.time()
            with self._lock:
                self._last_frame_time = ts
                self._last_valid_pct = valid_pct
                self._frame_count += 1

            frame = DepthFrame(depth=depth, confidence=confidence, timestamp=ts)

            # 非阻塞入队，丢弃旧帧，保留最新
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(frame)

    def _temporal_filter_depth(self, depth: np.ndarray) -> np.ndarray:
        if self._median_window <= 1:
            return depth

        self._depth_hist.append(depth.astype(np.float32, copy=True))
        if len(self._depth_hist) <= 1:
            return depth

        stack = np.stack(self._depth_hist, axis=0)
        stack = np.where(stack > 0.0, stack, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            filtered = np.nanmedian(stack, axis=0)
        return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get_depth_frame(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def get_status(self, now: float = None):
        with self._lock:
            last_frame_time = self._last_frame_time
            valid_pct = self._last_valid_pct
            frame_count = self._frame_count
            connected = self._connected
        if now is None:
            now = time.time()
        age_s = None if last_frame_time is None else max(0.0, now - last_frame_time)
        return {
            'connected': connected,
            'age_s': age_s,
            'valid_pct': valid_pct,
            'frame_count': frame_count,
        }

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════
# MotionEstimator：从姿态历史计算角速度 / 线速度，供 OccupancyMapper 做运动门控
# ══════════════════════════════════════════════════════════════════
class MotionEstimator:
    """
    维护一个短时位姿历史，用最早帧和最新帧的差分估计 ω 和 v，
    一阶 IIR (α=0.4) 平滑后输出 MotionState。
    """

    def __init__(self, window: int = 5, alpha: float = 0.4):
        self._hist   = deque(maxlen=window)
        self._alpha  = alpha
        self._yaw_rate = 0.0
        self._speed    = 0.0

    def update(self, t: float, pose: tuple):
        from occupancy_mapper import MotionState
        x, y, _z, yaw = pose
        self._hist.append((t, x, y, yaw))
        if len(self._hist) >= 2:
            t0, x0, y0, yaw0 = self._hist[0]
            t1, x1, y1, yaw1 = self._hist[-1]
            dt = max(t1 - t0, 1e-6)
            raw_omega = angular_diff_deg(yaw1, yaw0) / dt
            raw_speed = math.hypot(x1 - x0, y1 - y0) / dt   # cm/s
            a = self._alpha
            self._yaw_rate = a * raw_omega + (1 - a) * self._yaw_rate
            self._speed    = a * raw_speed  + (1 - a) * self._speed
        return MotionState(yaw_rate_deg_s=self._yaw_rate,
                           speed_cm_s=self._speed)


# ══════════════════════════════════════════════════════════════════
# Bresenham 直线算法（segment_is_free / validate_plan 仍在此文件内使用）
# ══════════════════════════════════════════════════════════════════
def _bresenham(u0, v0, u1, v1):
    cells = []
    dx, dy = abs(u1 - u0), abs(v1 - v0)
    sx = 1 if u1 >= u0 else -1
    sy = 1 if v1 >= v0 else -1
    err = dx - dy
    u, v = u0, v0
    while True:
        cells.append((u, v))
        if u == u1 and v == v1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            u   += sx
        if e2 < dx:
            err += dx
            v   += sy
    return cells


def segment_is_free(start_uv, end_uv, combined: np.ndarray,
                    skip_start_px: int = 1) -> bool:
    H, W = combined.shape
    ray = _bresenham(start_uv[0], start_uv[1], end_uv[0], end_uv[1])
    for i, (u, v) in enumerate(ray):
        if i < skip_start_px:
            continue
        if not (0 <= v < H and 0 <= u < W and combined[v, u]):
            return False
    return True


def analyze_depth_scan(depth: np.ndarray, cfg_depth: dict):
    """
    从当前深度图提取近场扇区距离。
    返回 front/left/right 的保守距离估计（米）以及 unknown_ratio（无效像素占比）。
    unknown_ratio > 0.7 表示该扇区深度覆盖率极低，调用方应按保守距离处理。
    """
    if depth is None:
        return None
    H, W = depth.shape
    r_min = cfg_depth['min_range_m']
    r_max = cfg_depth['max_range_m']
    v_lo = int(cfg_depth['row_band_frac'][0] * H)
    v_hi = int(cfg_depth['row_band_frac'][1] * H)
    c_band = cfg_depth.get('col_band_frac', [0.0, 1.0])
    c_lo = int(c_band[0] * W)
    c_hi = int(c_band[1] * W)
    if v_hi <= v_lo or c_hi <= c_lo:
        return None

    band = depth[v_lo:v_hi, c_lo:c_hi]
    width = band.shape[1]
    slices = {
        'left':  (0, max(1, int(width * 0.40))),
        'front': (max(0, int(width * 0.35)), min(width, int(width * 0.65))),
        'right': (min(width - 1, int(width * 0.60)), width),
    }

    min_valid = int(cfg_depth.get('sector_min_valid_px', 80))
    pct = float(cfg_depth.get('sector_percentile', 15.0))
    near_r = float(cfg_depth.get('near_obstacle_m', 2.0))
    near_min_valid = int(cfg_depth.get(
        'near_sector_min_valid_px',
        max(12, min_valid // 6)))
    sectors = {}
    for name, (a, b) in slices.items():
        vals = band[:, a:b]
        total_px = vals.size
        valid = vals[(vals > r_min) & (vals < r_max)]
        unknown_ratio = 1.0 - valid.size / max(total_px, 1)
        if valid.size >= min_valid:
            sectors[name] = {
                'distance_m': float(np.percentile(valid, pct)),
                'valid_px': int(valid.size),
                'unknown_ratio': float(unknown_ratio),
                'seen': True,
            }
            continue

        near_valid = valid[valid < near_r]
        if near_valid.size >= near_min_valid:
            sectors[name] = {
                'distance_m': float(np.percentile(near_valid, min(pct, 25.0))),
                'valid_px': int(valid.size),
                'unknown_ratio': float(unknown_ratio),
                'seen': True,
            }
        else:
            sectors[name] = {
                'distance_m': float('inf'),
                'valid_px': int(valid.size),
                'unknown_ratio': float(unknown_ratio),
                'seen': False,
            }

    return sectors


def _sector_distance_for_rel(scan, rel_deg: float) -> float:
    if scan is None:
        return float('inf')
    if rel_deg < -25.0:
        return scan['left']['distance_m']
    if rel_deg > 25.0:
        return scan['right']['distance_m']
    return scan['front']['distance_m']


def choose_control_target(auv_x: float, auv_y: float, auv_z: float,
                          auv_yaw: float,
                          nominal_target,
                          z_fixed: float,
                          depth_scan,
                          combined_safe: np.ndarray,
                          transformer: CoordinateTransformer,
                          cfg_plan: dict):
    """
    在发送 SetPose 前加一层局部避障。
    正常时只向 waypoint 迈短步；近障或短路径被占用时，选择侧绕/后退候选点。
    """
    auv_uv = transformer.world_to_pixel(auv_x, auv_y)
    goal_yaw = compute_yaw((auv_x, auv_y, auv_z), nominal_target)
    dist_to_wp = math.hypot(nominal_target[0] - auv_x, nominal_target[1] - auv_y)

    local_step = float(cfg_plan.get('local_step_cm', 120.0))
    min_step = float(cfg_plan.get('min_local_step_cm', 60.0))
    hard_m = float(cfg_plan.get('emergency_stop_m', 1.8))
    caution_m = float(cfg_plan.get('caution_range_m', 3.2))

    nominal_uv = transformer.world_to_pixel(
        nominal_target[0], nominal_target[1])
    nominal_ok = segment_is_free(
        auv_uv, nominal_uv, combined_safe, skip_start_px=2)

    step = min(max(min_step, dist_to_wp), local_step)
    direct_x = auv_x + step * math.cos(math.radians(goal_yaw))
    direct_y = auv_y + step * math.sin(math.radians(goal_yaw))
    direct_uv = transformer.world_to_pixel(direct_x, direct_y)
    direct_ok = nominal_ok if dist_to_wp <= step else segment_is_free(
        auv_uv, direct_uv, combined_safe, skip_start_px=2)

    front_m = depth_scan['front']['distance_m'] if depth_scan else float('inf')
    front_seen = False
    if depth_scan:
        unknown_ratio_th = float(cfg_plan.get('unknown_sector_ratio', 0.85))
        unknown_caution_factor = float(
            cfg_plan.get('unknown_sector_caution_factor', 0.9))
        depth_scan = {k: dict(v) for k, v in depth_scan.items()}
        for _sec in ('front', 'left', 'right'):
            _info = depth_scan[_sec]
            _seen = bool(_info.get('seen', False))
            _unknown = (not _seen and _info.get('unknown_ratio', 0.0) > unknown_ratio_th)
            _info['unknown'] = _unknown
            if _unknown:
                # 未知区域不再直接当成“2m 内必有障碍”，只作为偏保守候选。
                _info['distance_m'] = min(
                    _info['distance_m'], caution_m * unknown_caution_factor)
        front_m = depth_scan['front']['distance_m']
        front_seen = bool(depth_scan['front'].get('seen', False))
    must_avoid = front_seen and front_m < hard_m
    # Only activate VFH when range sensor confirms an obstacle — not on straight-line
    # segment checks, which conflict with TEB's curved obstacle-avoiding paths.
    should_avoid = must_avoid or (front_seen and front_m < caution_m)

    if not should_avoid:
        return nominal_target[0], nominal_target[1], z_fixed, goal_yaw, 'track'

    left_m = depth_scan['left']['distance_m'] if depth_scan else float('inf')
    right_m = depth_scan['right']['distance_m'] if depth_scan else float('inf')
    if depth_scan:
        left_pref = left_m if depth_scan['left'].get('seen', False) else caution_m * 0.8
        right_pref = right_m if depth_scan['right'].get('seen', False) else caution_m * 0.8
    else:
        left_pref = left_m
        right_pref = right_m
    prefer_sign = -1.0 if left_pref >= right_pref else 1.0
    rel_candidates = [
        prefer_sign * 70.0, -prefer_sign * 70.0,
        prefer_sign * 105.0, -prefer_sign * 105.0,
        0.0, 145.0, -145.0, 180.0,
    ]
    if not must_avoid:
        rel_candidates = [0.0, prefer_sign * 35.0, -prefer_sign * 35.0] + rel_candidates

    best = None
    for rel in rel_candidates:
        cand_yaw = normalize_angle_deg(auv_yaw + rel)
        sector_m = _sector_distance_for_rel(depth_scan, rel)
        clearance_step = local_step
        if math.isfinite(sector_m):
            clearance_step = max(min_step, (sector_m - hard_m * 0.55) * 100.0)
        if abs(rel) > 130.0:
            clearance_step = float(cfg_plan.get('reverse_dist_cm', 100.0))
        cand_step = min(local_step, clearance_step)
        if must_avoid and abs(rel) < 25.0:
            cand_step = min_step

        cx = auv_x + cand_step * math.cos(math.radians(cand_yaw))
        cy = auv_y + cand_step * math.sin(math.radians(cand_yaw))
        cuv = transformer.world_to_pixel(cx, cy)
        if not segment_is_free(auv_uv, cuv, combined_safe, skip_start_px=2):
            continue

        align = math.cos(math.radians(angular_diff_deg(cand_yaw, goal_yaw)))
        clearance_score = min(sector_m, caution_m) if math.isfinite(sector_m) else caution_m
        turn_penalty = abs(normalize_angle_deg(rel)) / 180.0
        score = 2.0 * align + 0.8 * clearance_score - 0.5 * turn_penalty
        if best is None or score > best[0]:
            best = (score, cx, cy, cand_yaw)

    if best is not None:
        reason = 'emergency' if must_avoid else 'avoid'
        return best[1], best[2], z_fixed, best[3], reason

    # 极近障碍时所有候选都被堵：原地保持，等待重规划，避免顶墙激进侧绕。
    if must_avoid and front_m < hard_m * 0.7:
        return auv_x, auv_y, z_fixed, auv_yaw, 'hold'

    # 其余情况：短距离后退给规划器争取时间。
    back_yaw = normalize_angle_deg(auv_yaw + 180.0)
    back_step = float(cfg_plan.get('reverse_dist_cm', 100.0))
    bx = auv_x + back_step * math.cos(math.radians(back_yaw))
    by = auv_y + back_step * math.sin(math.radians(back_yaw))
    return bx, by, z_fixed, back_yaw, 'reverse'


# ══════════════════════════════════════════════════════════════════
# 路径前视跟踪目标
# ══════════════════════════════════════════════════════════════════
def select_tracking_waypoint(auv_x: float, auv_y: float, auv_z: float,
                             waypoints, cursor: int,
                             cfg_plan: dict):
    """
    不直接盯当前 waypoint，而是沿路径挑选一个前视目标点，
    减少稠密 waypoint 下的左右抖动和频繁换向。
    """
    if not waypoints or cursor >= len(waypoints):
        return None

    lookahead_cm = float(cfg_plan.get(
        'track_lookahead_cm',
        max(cfg_plan.get('local_step_cm', 120.0) * 1.5,
            cfg_plan.get('waypoint_reach_cm', 100.0) * 2.0)))

    prev = (auv_x, auv_y, auv_z)
    acc = 0.0
    candidate = waypoints[cursor]
    for wp in waypoints[cursor:]:
        seg = math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        acc += seg
        candidate = wp
        if acc >= lookahead_cm:
            break
        prev = wp
    return candidate


# ══════════════════════════════════════════════════════════════════
# 路径有效性检查
# ══════════════════════════════════════════════════════════════════
def validate_plan(waypoints, cursor: int,
                  combined: np.ndarray,   # bool，True=可通行
                  K: int,
                  transformer: CoordinateTransformer,
                  auv_pix=None) -> bool:
    """检查从 AUV 当前位置起的前 K 个 waypoint 路段是否畅通。
    使用 Bresenham 扫描相邻 waypoint 之间的所有像素格，而非仅检查端点。
    """
    H, W = combined.shape

    def _blocked(u, v):
        return 0 <= v < H and 0 <= u < W and not combined[v, u]

    # 起始像素：AUV 当前位置；若未提供则从第一个 waypoint 开始
    prev = auv_pix if auv_pix is not None else \
        transformer.world_to_pixel(waypoints[cursor][0], waypoints[cursor][1])

    for wp in waypoints[cursor: cursor + K]:
        u, v = transformer.world_to_pixel(wp[0], wp[1])
        for (ru, rv) in _bresenham(prev[0], prev[1], u, v):
            if _blocked(ru, rv):
                return False
        prev = (u, v)
    return True


# ══════════════════════════════════════════════════════════════════
# 逃脱目标选取（停滞恢复）
# ══════════════════════════════════════════════════════════════════
def find_escape_target(auv_x: float, auv_y: float,
                       goal_x: float, goal_y: float,
                       combined: np.ndarray,
                       transformer: CoordinateTransformer,
                       dist_cm: float = 500.0):
    """
    当鱼停滞时，在目标方向的垂直 / 斜向寻找一个可直达的逃脱点。
    依次尝试 ±90°、±120°、±150°、180° 共 7 个方向；每个方向先试全距，再试 60% 距。
    返回 (x_cm, y_cm) 或 None。
    """
    goal_angle = math.atan2(goal_y - auv_y, goal_x - auv_x)
    H, W = combined.shape
    au, av = transformer.world_to_pixel(auv_x, auv_y)

    for deg in [90, -90, 120, -120, 150, -150, 180]:
        angle = goal_angle + math.radians(deg)
        for scale in [1.0, 0.6]:
            tx = auv_x + math.cos(angle) * dist_cm * scale
            ty = auv_y + math.sin(angle) * dist_cm * scale
            tu, tv = transformer.world_to_pixel(tx, ty)
            if not (0 <= tv < H and 0 <= tu < W and combined[tv, tu]):
                continue
            # 从 AUV 到逃脱点的 Bresenham 路段全部可通行
            if segment_is_free((au, av), (tu, tv), combined, skip_start_px=2):
                return (tx, ty)
    return None


# ══════════════════════════════════════════════════════════════════
# Hybrid A* 在线后处理
# ══════════════════════════════════════════════════════════════════
def build_hybrid_astar_config(grid_config: dict, nav_cfg: dict) -> HybridAStarConfig:
    """
    允许 navigate 在线阶段从 navigation_config.json 覆盖 Hybrid A* 参数。
    若 nav_cfg 中没有 hybrid_astar 段，则回退到 grid_config 默认值。
    """
    merged_cfg = copy.deepcopy(grid_config)
    nav_ha = nav_cfg.get('hybrid_astar')
    if isinstance(nav_ha, dict):
        merged_cfg.setdefault('hybrid_astar', {}).update(nav_ha)
    return HybridAStarConfig.from_config(merged_cfg)


# ══════════════════════════════════════════════════════════════════
# Hybrid A* 重规划
# ══════════════════════════════════════════════════════════════════
def run_replan(auv_x: float, auv_y: float, nav_yaw_deg: float,
               goal_x: float, goal_y: float, z_fixed: float,
               combined: np.ndarray,
               grid_config: dict,
               transformer: CoordinateTransformer,
               nav_cfg: dict):
    """
    在 combined（True=可通行）上执行 Hybrid A*，返回 List[(x_cm,y_cm,z_cm)] 或 None。
    """
    combined = combined.copy()
    H, W = combined.shape

    # 起点/终点强制保通：keepout 必须 > inflate 半径，否则 ConfigurationSpace
    # 膨胀时会把 keepout 区域外侧的障碍"填回"到起点附近，导致 A* 开局即无路可走。
    # 用 inflation_radius_m + safety_margin_m 的完整膨胀量再加 2 格冗余。
    keepout_px = int(math.ceil(
        (grid_config['planning']['inflation_radius_m']
         + grid_config['planning'].get('safety_margin_m', 0.0))
        * 100.0 / grid_config['resolution']['cm_per_pixel'])) + 2
    su, sv = transformer.world_to_pixel(auv_x, auv_y)
    combined[max(0, sv - keepout_px):min(H, sv + keepout_px + 1),
             max(0, su - keepout_px):min(W, su + keepout_px + 1)] = True

    # 终点周围强制保通
    gu, gv = transformer.world_to_pixel(goal_x, goal_y)
    combined[max(0, gv - keepout_px):min(H, gv + keepout_px + 1),
             max(0, gu - keepout_px):min(W, gu + keepout_px + 1)] = True

    cspace = ConfigurationSpace(combined, grid_config)
    cspace.inflate_obstacles()
    cspace.compute_distance_field()

    ha_cfg  = build_hybrid_astar_config(grid_config, nav_cfg)
    planner = HybridAStarPlanner(cspace, ha_cfg, transformer)

    start_pix   = transformer.world_to_pixel(auv_x, auv_y)
    goal_pix    = transformer.world_to_pixel(goal_x, goal_y)
    start_theta = math.radians(nav_yaw_deg)

    raw_path = planner.search(start_pix, goal_pix, start_theta)
    if raw_path is None:
        return None

    interval  = nav_cfg['planner']['resample_interval_cm']
    waypoints = postprocess_hybrid_path(
        raw_path, cspace, transformer, ha_cfg, z_fixed, interval)
    if not waypoints:
        return None
    return waypoints  # List[(x_cm, y_cm, z_cm)]


def run_replan_with_fallback(auv_x, auv_y, nav_yaw_deg,
                              goal_x, goal_y, z_fixed,
                              combined, grid_config, transformer, nav_cfg):
    """先用正常参数规划；失败后放宽 safety_margin 再试一次。"""
    wp = run_replan(auv_x, auv_y, nav_yaw_deg,
                    goal_x, goal_y, z_fixed,
                    combined, grid_config, transformer, nav_cfg)
    if wp is not None:
        return wp
    grid_cfg2 = copy.deepcopy(grid_config)
    if 'safety_margin_m' in grid_cfg2.get('planning', {}):
        grid_cfg2['planning']['safety_margin_m'] *= 0.5
        print('[NAV] fallback：safety_margin 减半重试规划')
        wp = run_replan(auv_x, auv_y, nav_yaw_deg,
                        goal_x, goal_y, z_fixed,
                        combined, grid_cfg2, transformer, nav_cfg)
    return wp


# ══════════════════════════════════════════════════════════════════
# SetPose 发送
# ══════════════════════════════════════════════════════════════════
def send_setpose(ctrl_sock: socket.socket,
                 x_cm: float, y_cm: float, z_cm: float,
                 nav_yaw_deg: float):
    cmd = f'SetPose:{x_cm:.1f},{y_cm:.1f},{z_cm:.1f},0,{nav_yaw_deg:.2f},0\n'
    ctrl_sock.sendall(cmd.encode('utf-8'))


def send_setvelocity(ctrl_sock: socket.socket,
                     vx_cm_s: float, vy_cm_s: float, vz_cm_s: float):
    """世界系线速度指令，单位 cm/s。格式：SetVelocity:vx,vy,vz"""
    cmd = f'SetVelocity:{vx_cm_s:.2f},{vy_cm_s:.2f},{vz_cm_s:.2f}\n'
    ctrl_sock.sendall(cmd.encode('utf-8'))


def send_settwist(ctrl_sock: socket.socket,
                  vx_b: float, vy_b: float, vz_b: float,
                  pitch_rate: float, yaw_rate: float, roll_rate: float):
    """鱼体局部坐标系速度+角速度指令。
    线速度单位 cm/s，角速度单位 deg/s。格式：SetTwist:vx,vy,vz,pr,yr,rr"""
    cmd = (f'SetTwist:{vx_b:.2f},{vy_b:.2f},{vz_b:.2f},'
           f'{pitch_rate:.2f},{yaw_rate:.2f},{roll_rate:.2f}\n')
    ctrl_sock.sendall(cmd.encode('utf-8'))


def _world_to_body(vx_w: float, vy_w: float, yaw_deg: float):
    """将世界系 2D 速度旋转到鱼体坐标系 (前向, 侧向)。"""
    yr = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yr), math.sin(yr)
    return cos_y * vx_w + sin_y * vy_w, -sin_y * vx_w + cos_y * vy_w


def pose_target_to_world_twist(auv_x: float, auv_y: float, auv_yaw: float,
                               tgt_x: float, tgt_y: float,
                               cfg_teb: dict):
    """根据目标位姿生成世界系平面速度与 yaw 角速度，用于 velocity/twist 模式。"""
    yaw_cmd = math.degrees(math.atan2(tgt_y - auv_y, tgt_x - auv_x))
    max_speed = float(cfg_teb.get('max_speed_cm_s', 150.0))
    max_omega = float(cfg_teb.get('max_yaw_rate_deg_s', 60.0))
    dist = math.hypot(tgt_x - auv_x, tgt_y - auv_y)
    yaw_rad = math.radians(yaw_cmd)
    if dist > 1.0:
        vx = max_speed * math.cos(yaw_rad)
        vy = max_speed * math.sin(yaw_rad)
    else:
        vx, vy = 0.0, 0.0
    yaw_err = normalize_angle_deg(yaw_cmd - auv_yaw)
    omega = max(-max_omega, min(max_omega, yaw_err * 3.0))
    return (vx, vy, 0.0, omega)


# ══════════════════════════════════════════════════════════════════
# 调试可视化
# ══════════════════════════════════════════════════════════════════
def draw_debug(static_vis: np.ndarray,   # float32，0=障碍，1=自由
               L: np.ndarray,
               l_thresh: float,
               waypoints, cursor: int,
               auv_x: float, auv_y: float, nav_yaw_deg: float,
               goal_x: float, goal_y: float,
               transformer: CoordinateTransformer,
               state: str, replan_total: int,
               replan_streak: int,
               local_reason: str = 'track',
               combined_safe: np.ndarray = None,
               map_stats: 'MapStats | None' = None) -> np.ndarray:
    base = (static_vis * 200).astype(np.uint8)
    vis  = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # 蓝色半透明 = 动态膨胀后的不可通行格（规划器视角的禁区）
    if combined_safe is not None:
        unsafe_mask = ~combined_safe & static_vis.astype(bool)
        overlay = vis.copy()
        overlay[unsafe_mask] = (180, 60, 0)   # BGR: 深蓝
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    # 红色 = 动态障碍（log-odds 超阈值原始格，未膨胀）
    vis[L > l_thresh] = (0, 0, 200)

    # 绿色路径
    if waypoints and len(waypoints) > cursor:
        pts = [transformer.world_to_pixel(wp[0], wp[1])
               for wp in waypoints[cursor:]]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], (0, 200, 0), 1)

    # 黄色 ★ 终点
    gu, gv = transformer.world_to_pixel(goal_x, goal_y)
    cv2.drawMarker(vis, (gu, gv), (0, 215, 255), cv2.MARKER_STAR, 12, 2)

    # 蓝色箭头 AUV
    au, av = transformer.world_to_pixel(auv_x, auv_y)
    phi = math.radians(nav_yaw_deg)
    eu  = int(au + 8 * math.cos(phi))
    ev  = int(av + 8 * math.sin(phi))
    cv2.arrowedLine(vis, (au, av), (eu, ev), (255, 80, 0), 2, tipLength=0.4)

    # HUD
    dist = math.hypot(auv_x - goal_x, auv_y - goal_y)
    hud  = [
        f'State: {state}   Replans: {replan_total}  Streak: {replan_streak}',
        f'AUV ({auv_x:.0f}, {auv_y:.0f})  yaw={nav_yaw_deg:.1f}',
        f'dist_goal={dist:.0f}cm  WP {cursor}/{len(waypoints) if waypoints else 0}',
        f'Local: {local_reason}',
    ]
    if map_stats is not None:
        hud.append(
            f'gated={map_stats.gate_pct:.0f}%'
            f'  far_rej={map_stats.far_rej_pct:.0f}%'
            f'  conf={map_stats.last_avg_pixel_conf:.2f}')
    for i, txt in enumerate(hud):
        cv2.putText(vis, txt, (4, 14 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return vis


# ══════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════
def main():
    # ── 加载配置（grid 参数已合并进 navigation_config.json）───────────
    with open(NAV_CFG_PATH) as f: nav_cfg = json.load(f)
    grid_config = nav_cfg['grid']

    sp        = nav_cfg['start_pose']
    gp        = nav_cfg['goal_pose']
    z_fixed   = sp['z']
    goal_x,   goal_y = gp['x'], gp['y']

    cfg_depth    = nav_cfg['depth']
    cfg_log      = nav_cfg['log_odds']
    cfg_plan     = nav_cfg['planner']
    cfg_timing   = nav_cfg['timing']
    ue_tcp       = nav_cfg['ue_tcp']
    cfg_teb      = nav_cfg.get('teb', {})
    control_mode = ue_tcp.get('control_mode', 'pose')

    # 启动时校验目标点是否在占用图范围内
    _wb = grid_config['world_bounds']
    if not (_wb['x_min'] <= goal_x <= _wb['x_max'] and
            _wb['y_min'] <= goal_y <= _wb['y_max']):
        print(f'[NAV] ✗ 错误：目标点 ({goal_x}, {goal_y}) cm 不在占用图范围内！'
              f" x=[{_wb['x_min']}, {_wb['x_max']}] y=[{_wb['y_min']}, {_wb['y_max']}]")
        raise SystemExit(1)

    rate_hz       = cfg_timing['control_rate_hz']
    dt            = 1.0 / rate_hz
    pose_init_to  = cfg_timing.get('pose_init_timeout_s', 30.0)
    pose_to       = cfg_timing['pose_timeout_s']
    reconnect_s   = cfg_timing['pose_reconnect_s']
    frame_to      = cfg_timing['frame_timeout_s']
    l_thresh    = cfg_log['l_thresh']

    # ── 初始化地图与运动估计器 ───────────────────────────────────────
    img_cfg = grid_config['image_size']
    H, W   = img_cfg['height'], img_cfg['width']
    static_vis = np.ones((H, W), dtype=np.float32)  # 可视化用（全白）

    transformer = CoordinateTransformer(grid_config)
    mapper      = OccupancyMapper(grid_config, nav_cfg, transformer)
    motion_est  = MotionEstimator()

    # ── TEB 局部规划器 ───────────────────────────────────────────────
    ha_cfg_for_teb = build_hybrid_astar_config(grid_config, nav_cfg)
    teb_planner = TEBLocalPlanner(
        cfg_teb, ha_cfg_for_teb, transformer,
        grid_config['resolution']['cm_per_pixel'])

    # ── 启动后台线程 ─────────────────────────────────────────────────
    pose_listener = PoseListener(
        ue_tcp['pose_host'], ue_tcp['pose_port'], reconnect_s)
    frame_worker  = FrameWorker(
        ue_tcp['frame_host'], ue_tcp['frame_port'], frame_to,
        median_window=int(cfg_depth.get('temporal_median_frames', 1)))
    pose_listener.start()
    frame_worker.start()

    # ── 控制通道（重试直到 UE 启动） ────────────────────────────────
    ctrl_host = ue_tcp['control_host']
    ctrl_port = ue_tcp['control_port']
    ctrl_sock = None
    print(f'[NAV] 等待 UE 控制通道 {ctrl_host}:{ctrl_port} ...')
    while ctrl_sock is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((ctrl_host, ctrl_port))
            s.settimeout(2.0)
            ctrl_sock = s
            print(f'[NAV] 控制通道已连接 {ctrl_host}:{ctrl_port}')
        except (ConnectionRefusedError, OSError, socket.timeout):
            s.close()
            print(f'[NAV] 控制通道未就绪，1s 后重试...')
            time.sleep(1.0)

    # ── 可视化窗口 ───────────────────────────────────────────────────
    cv2.namedWindow('Navigate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Navigate', 800, 480)

    # ── 录像（保存到 result/navigate_<时间戳>.mp4）────────────────────
    _vwriter = None
    if RECORD_VIDEO:
        _video_path = os.path.join(_BASE, 'result',
                                   time.strftime('navigate_%Y%m%d_%H%M%S.mp4'))
        _fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        _vwriter = cv2.VideoWriter(_video_path, _fourcc, rate_hz, (W, H))
        if _vwriter.isOpened():
            print(f'[NAV] 录像已开启 → {_video_path}')
        else:
            print('[NAV] 警告：VideoWriter 初始化失败，不录像')
            _vwriter = None

    # ── 状态机变量 ───────────────────────────────────────────────────
    state        = IDLE
    waypoints    = []     # List[(x_cm, y_cm, z_cm)]
    cursor       = 0
    replan_total = 0
    replan_streak = 0
    invalid_plan_streak = 0
    last_replan  = 0.0
    last_pose_ok = None   # 首次收到 pose 的时间
    # 进度监控
    min_dist_seen      = float('inf')
    last_progress_time = None
    stall_timeout_s    = cfg_plan.get('stall_timeout_s', 15.0)
    replan_confirm_frames = max(1, int(cfg_plan.get('replan_confirm_frames', 3)))
    last_depth_scan    = None
    last_depth_time    = 0.0
    last_local_reason  = 'track'
    last_map_stats     = MapStats()
    _log_frame = 0          # 帧计数，每 rate_hz 帧打印一次详细日志

    try:
        while state not in (REACHED_GOAL, FAILED):
            t0 = time.time()

            # ──────────────────────── IDLE ────────────────────────
            if state == IDLE:
                pose = pose_listener.get_pose()
                if pose is None:
                    if last_pose_ok is None:
                        last_pose_ok = time.time()  # 开始计时
                    elif (time.time() - last_pose_ok) > pose_init_to:
                        print('[NAV] 等待位姿超时 → FAILED')
                        state = FAILED
                    time.sleep(0.1)
                    continue

                auv_x, auv_y, auv_z, auv_yaw = pose
                auv_u, auv_v = transformer.world_to_pixel(auv_x, auv_y)
                goal_u, goal_v = transformer.world_to_pixel(goal_x, goal_y)
                print(f'[NAV] 首帧 pose: world=({auv_x:.0f}, {auv_y:.0f}) pixel=({auv_u}, {auv_v})')
                print(f'[NAV] 目标点:  world=({goal_x:.0f}, {goal_y:.0f}) pixel=({goal_u}, {goal_v})')
                if not (0 <= auv_u < transformer.width and 0 <= auv_v < transformer.height):
                    print(f'[NAV] ⚠ 警告：AUV 出生点在占用图范围之外！'
                          f' 图范围 x=[{transformer.x_min}, {transformer.x_max}]'
                          f' y=[{transformer.y_min}, {transformer.y_max}]')

                # 初始规划：全图乐观（mapper 刚初始化，L=0，combined_raw 全为 True）
                _init_views = mapper.get_views((auv_u, auv_v))
                wp = run_replan_with_fallback(
                    auv_x, auv_y, auv_yaw,
                    goal_x, goal_y, z_fixed,
                    _init_views.combined_raw, grid_config, transformer, nav_cfg)
                if wp is None:
                    print('[NAV] 初始规划失败 → FAILED')
                    state = FAILED
                    continue
                waypoints   = wp
                cursor      = 0
                replan_streak = 0
                invalid_plan_streak = 0
                last_pose_ok = time.time()
                print(f'[NAV] 初始路径 {len(waypoints)} 个 waypoint → NAV')
                print(f'[NAV] 前5个 waypoint（世界坐标 cm）:')
                for _i, _w in enumerate(waypoints[:5]):
                    _da = math.degrees(math.atan2(
                        _w[1] - auv_y, _w[0] - auv_x))
                    print(f'       [{_i}] ({_w[0]:7.0f}, {_w[1]:7.0f})'
                          f'  AUV→WP 方向={_da:.1f}°  '
                          f'  (目标方向='
                          f'{math.degrees(math.atan2(goal_y-auv_y, goal_x-auv_x)):.1f}°)')
                state = NAV
                continue

            # ──────────────────────── NAV ─────────────────────────
            pose        = pose_listener.get_pose()
            depth_frame = frame_worker.get_depth_frame()

            if pose is None:
                if last_pose_ok and (time.time() - last_pose_ok) > pose_to:
                    print('[NAV] 位姿丢失超时 → FAILED')
                    state = FAILED
                time.sleep(dt)
                continue
            last_pose_ok = time.time()

            auv_x, auv_y, auv_z, auv_yaw = pose
            auv_u, auv_v = transformer.world_to_pixel(auv_x, auv_y)
            now = time.time()
            frame_status = frame_worker.get_status(now)
            depth_age_s = frame_status['age_s']
            depth_valid_pct = frame_status['valid_pct']
            depth_connected = frame_status['connected']
            depth_stale_timeout_s = float(cfg_depth.get('stale_timeout_s', 1.0))
            hold_on_stale_after_s = float(
                cfg_depth.get('hold_on_stale_after_s', depth_stale_timeout_s * 1.5))

            # 运动估计（用于 mapper 运动门控）
            motion = motion_est.update(now, pose)

            # 深度投影 + range-aware log-odds 更新（包含衰减、AUV 脚印清除）
            if depth_frame is not None:
                last_depth_scan = analyze_depth_scan(depth_frame.depth, cfg_depth)
                last_depth_time = now
                last_map_stats = mapper.update(depth_frame, pose, motion)
            depth_scan = last_depth_scan if (
                last_depth_scan is not None
                and (now - last_depth_time) < depth_stale_timeout_s) else None

            # 构建合并地图（True = 可通行）
            views = mapper.get_views((auv_u, auv_v))
            combined           = views.combined_raw
            combined_safe      = views.combined_safe
            combined_safe_local = views.combined_safe_local

            # 到达终点
            dist_goal = math.hypot(auv_x - goal_x, auv_y - goal_y)
            if dist_goal < cfg_plan['goal_tolerance_cm']:
                print(f'[NAV] 到达终点 dist={dist_goal:.1f}cm → REACHED_GOAL')
                state = REACHED_GOAL
                break

            # 进度监控：长时间无进展则插入一个可直达逃脱点，保留已观测障碍
            if last_progress_time is None:
                last_progress_time = now
                min_dist_seen = dist_goal
            if dist_goal < min_dist_seen - 50.0:
                min_dist_seen = dist_goal
                last_progress_time = now
                replan_streak = 0
            elif (now - last_progress_time) > stall_timeout_s:
                print(f'[NAV] 连续 {stall_timeout_s:.0f}s 无进展 (dist={dist_goal:.0f}cm) → 寻找逃脱路径')
                escape = find_escape_target(
                    auv_x, auv_y, goal_x, goal_y,
                    combined_safe_local, transformer,
                    cfg_plan.get('escape_dist_cm', 500.0))
                if escape is not None:
                    print(f'[NAV] 逃脱目标 → ({escape[0]:.0f}, {escape[1]:.0f})')
                    # 只保留逃脱点本身，不附加旧路径 tail。
                    # 旧路径是被障碍封堵的原因；到达逃脱点后由主循环自然触发重规划。
                    # 不重置 last_replan，让鱼实际走到逃脱点再规划，避免立刻被覆盖。
                    waypoints = [(escape[0], escape[1], z_fixed)]
                    cursor = 0
                    invalid_plan_streak = 0
                min_dist_seen = dist_goal
                last_progress_time = now
                replan_streak = 0

            # waypoint 推进
            while cursor < len(waypoints):
                wp = waypoints[cursor]
                if math.hypot(auv_x - wp[0], auv_y - wp[1]) < cfg_plan['waypoint_reach_cm']:
                    cursor += 1
                else:
                    break
            cursor = min(cursor, len(waypoints) - 1) if waypoints else 0

            # 路径有效性校验：用 combined_safe（障碍膨胀后的通行图），
            # 不用 combined_safe_local（后者额外限制未观测区域 = 圆盘外全当障碍）。
            # combined_safe_local 用于 TEB/局部避障；全局路径校验只需知道
            # "是否有已确认障碍挡路"，不应因未扫描区域而误判无效。
            if depth_scan is None:
                plan_valid = True
                invalid_plan_streak = 0
            else:
                plan_valid = validate_plan(
                    waypoints, cursor, combined_safe,
                    cfg_plan['lookahead_waypoints'], transformer,
                    auv_pix=(auv_u, auv_v))
                if plan_valid:
                    invalid_plan_streak = 0
                else:
                    invalid_plan_streak += 1

            need_replan = invalid_plan_streak >= replan_confirm_frames
            if need_replan and (now - last_replan) > cfg_plan['replan_cooldown_s']:
                replan_total += 1
                replan_streak += 1
                print(
                    f'[NAV] REPLAN #{replan_total} '
                    f'(streak={replan_streak}, invalid={invalid_plan_streak})  '
                    f'dist={dist_goal:.0f}cm'
                )
                if replan_streak > cfg_plan['max_replans']:
                    print('[NAV] 连续无进展重规划次数超限 → FAILED')
                    state = FAILED
                    break
                # 重规划前清除 AUV 周围未确认的弱标记，保留已多帧确认的真障碍
                mapper.clear_around_auv((auv_u, auv_v))
                views = mapper.get_views((auv_u, auv_v))
                combined            = views.combined_raw
                combined_safe       = views.combined_safe
                combined_safe_local = views.combined_safe_local

                # 全局规划：乐观地图（仅剔除 log-odds 确认障碍），不受 observed_mask 约束；
                # 局部层 combined_safe_local 已负责阻止 AUV 闯入未观测盲区。
                new_wp = run_replan_with_fallback(
                    auv_x, auv_y, auv_yaw,
                    goal_x, goal_y, z_fixed,
                    combined, grid_config, transformer, nav_cfg)

                if new_wp is None:
                    # 二阶恢复：强制清除 AUV 周围已确认障碍（包含多帧积累的柱体检测），
                    # 重新构建地图后再试一次。适用于"鱼卡在柱子近旁反复振荡"场景。
                    print('[NAV] 重规划无解，扩大清除后重试...')
                    mapper.hard_clear_around_auv((auv_u, auv_v))
                    views = mapper.get_views((auv_u, auv_v))
                    combined            = views.combined_raw
                    combined_safe       = views.combined_safe
                    combined_safe_local = views.combined_safe_local
                    new_wp = run_replan_with_fallback(
                        auv_x, auv_y, auv_yaw,
                        goal_x, goal_y, z_fixed,
                        combined, grid_config, transformer, nav_cfg)

                if new_wp is None:
                    # 三阶恢复：后退 3 倍 reverse_dist，插入临时 waypoint；
                    # 让 log-odds 图随时间衰减，下一轮重规划有机会找到路径。
                    print('[NAV] 重规划仍无解，执行后退恢复（不终止）')
                    back_yaw  = normalize_angle_deg(auv_yaw + 180.0)
                    back_step = float(cfg_plan.get('reverse_dist_cm', 100.0)) * 3
                    back_x = auv_x + back_step * math.cos(math.radians(back_yaw))
                    back_y = auv_y + back_step * math.sin(math.radians(back_yaw))
                    waypoints = [(back_x, back_y, z_fixed)]
                    cursor    = 0
                    last_replan = now
                    invalid_plan_streak = 0
                    replan_streak = max(0, replan_streak - 2)  # 恢复不计入连续失败
                else:
                    # 规划成功，但连续多次规划仍无法前进（路径被立刻判无效）→
                    # 强制后退解套：让鱼实际离开障碍区，再从新位置规划。
                    force_backup_streak = int(
                        cfg_plan.get('force_backup_streak', 4))
                    if replan_streak >= force_backup_streak:
                        print(f'[NAV] 重规划连续 {replan_streak} 次路径仍被立刻封堵，'
                              f'强制后退解套')
                        mapper.hard_clear_around_auv((auv_u, auv_v))
                        back_yaw  = normalize_angle_deg(auv_yaw + 180.0)
                        back_step = float(cfg_plan.get('reverse_dist_cm', 100.0)) * 2
                        back_x = auv_x + back_step * math.cos(math.radians(back_yaw))
                        back_y = auv_y + back_step * math.sin(math.radians(back_yaw))
                        waypoints = [(back_x, back_y, z_fixed)]
                        cursor    = 0
                        last_replan = now
                        invalid_plan_streak = 0
                        replan_streak = 0
                    else:
                        waypoints   = new_wp
                        cursor      = 0
                        last_replan = now
                        invalid_plan_streak = 0

            # 发送指令：经 TEB 形变后的局部目标点
            if cursor < len(waypoints):
                if depth_age_s is None or depth_age_s > hold_on_stale_after_s:
                    last_local_reason = 'depth_stale'
                    try:
                        if control_mode == 'velocity':
                            send_setvelocity(ctrl_sock, 0.0, 0.0, 0.0)
                        elif control_mode == 'twist':
                            send_settwist(ctrl_sock, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        else:
                            send_setpose(ctrl_sock, auv_x, auv_y, z_fixed, auv_yaw)
                    except OSError as e:
                        print(f'[NAV] 控制连接断开：{e} → FAILED')
                        state = FAILED
                        break
                    _log_frame += 1
                    if _log_frame % max(1, int(rate_hz)) == 0:
                        print(
                            f'[NAV] t={_log_frame/rate_hz:5.1f}s'
                            f'  AUV=({auv_x:7.0f},{auv_y:7.0f}) yaw={auv_yaw:6.1f}°'
                            f'  dist_goal={dist_goal:6.0f}cm'
                            f'  depth=STALE age={_fmt_metric(depth_age_s, "s", 4, 2)}'
                            f' valid={_fmt_metric(depth_valid_pct, "%", 5, 1)}'
                            f' connected={depth_connected}'
                        )
                    vis = draw_debug(
                        static_vis, views.L, l_thresh,
                        waypoints, cursor,
                        auv_x, auv_y, auv_yaw,
                        goal_x, goal_y,
                        transformer, state, replan_total, replan_streak,
                        last_local_reason, combined_safe, last_map_stats)
                    cv2.imshow('Navigate', vis)
                    if _vwriter is not None:
                        _vwriter.write(vis)
                    cv2.waitKey(1)
                    elapsed = time.time() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
                    continue

                # 深度近场急停：front < emergency_stop_m 时原地保持，跳过 TEB
                _front_m = (depth_scan['front']['distance_m']
                            if depth_scan else float('inf'))
                _hard_m  = float(cfg_plan.get('emergency_stop_m', 2.5))
                if _front_m < _hard_m * 0.7:
                    last_local_reason = 'hold'
                    try:
                        if control_mode == 'velocity':
                            send_setvelocity(ctrl_sock, 0.0, 0.0, 0.0)
                        elif control_mode == 'twist':
                            send_settwist(ctrl_sock, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        else:
                            send_setpose(ctrl_sock, auv_x, auv_y, z_fixed, auv_yaw)
                    except OSError as e:
                        print(f'[NAV] 控制连接断开：{e} → FAILED')
                        state = FAILED
                        break
                    # 继续主循环（不走 TEB 分支）
                    vis = draw_debug(
                        static_vis, views.L, l_thresh,
                        waypoints, cursor,
                        auv_x, auv_y, auv_yaw,
                        goal_x, goal_y,
                        transformer, state, replan_total, replan_streak,
                        last_local_reason, combined_safe, last_map_stats)
                    cv2.imshow('Navigate', vis)
                    if _vwriter is not None:
                        _vwriter.write(vis)
                    cv2.waitKey(1)
                    elapsed = time.time() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
                    continue

                tgt_pose, tgt_twist = teb_planner.update(
                    waypoints, cursor,
                    (auv_x, auv_y, auv_z, auv_yaw),
                    combined_safe_local, dt)

                if tgt_pose is None:
                    # TEB 路径窗口耗尽：pose 模式回退跟踪最后 waypoint；
                    # velocity/twist 模式发零速让 AUV 原地停住。
                    wp = waypoints[cursor]
                    tgt_pose = (wp[0], wp[1], z_fixed,
                                compute_yaw((auv_x, auv_y, auv_z), wp))
                    tgt_twist = None

                nominal_pose = tgt_pose
                cmd_x, cmd_y, cmd_z, yaw_to_tgt, local_reason = choose_control_target(
                    auv_x, auv_y, auv_z, auv_yaw,
                    nominal_pose,
                    z_fixed,
                    depth_scan,
                    combined_safe_local,
                    transformer,
                    cfg_plan)
                # tgt_twist = (vx_world_cm_s, vy_world_cm_s, vz_cm_s, omega_deg_s)
                if local_reason == 'track':
                    last_local_reason = 'teb'
                else:
                    last_local_reason = local_reason
                    tgt_twist = pose_target_to_world_twist(
                        auv_x, auv_y, auv_yaw,
                        cmd_x, cmd_y,
                        cfg_teb)

                # ── 每秒打印一次导航诊断日志 ────────────────────────
                _log_frame += 1
                if _log_frame % max(1, int(rate_hz)) == 0:
                    _angle_to_goal = math.degrees(
                        math.atan2(goal_y - auv_y, goal_x - auv_x))
                    _wp = waypoints[cursor] if cursor < len(waypoints) else None
                    _angle_to_wp = math.degrees(
                        math.atan2(_wp[1] - auv_y, _wp[0] - auv_x)) if _wp else float('nan')
                    _angle_to_cmd = math.degrees(
                        math.atan2(cmd_y - auv_y, cmd_x - auv_x)) if (
                        abs(cmd_x - auv_x) + abs(cmd_y - auv_y) > 1) else float('nan')
                    print(
                        f'[NAV] t={_log_frame/rate_hz:5.1f}s'
                        f'  AUV=({auv_x:7.0f},{auv_y:7.0f}) yaw={auv_yaw:6.1f}°'
                        f'  dist_goal={dist_goal:6.0f}cm  angle_to_goal={_angle_to_goal:6.1f}°'
                    )
                    print(
                        f'           Depth age={_fmt_metric(depth_age_s, "s", 4, 2)}'
                        f'  valid={_fmt_metric(depth_valid_pct, "%", 5, 1)}'
                        f'  link={depth_connected}'
                        f'  front={_fmt_metric(depth_scan["front"]["distance_m"], "m") if depth_scan else "stale"}'
                        f'  left={_fmt_metric(depth_scan["left"]["distance_m"], "m") if depth_scan else "stale"}'
                        f'  right={_fmt_metric(depth_scan["right"]["distance_m"], "m") if depth_scan else "stale"}'
                    )
                    if _wp:
                        print(
                            f'           WP[{cursor}]=({_wp[0]:7.0f},{_wp[1]:7.0f})'
                            f'  dir={_angle_to_wp:6.1f}°'
                            f'  | TEB→({cmd_x:7.1f},{cmd_y:7.1f})'
                            f'  cmd_yaw={yaw_to_tgt:6.1f}°'
                            f'  cmd_dir={_angle_to_cmd:6.1f}°'
                            f'  mode={last_local_reason}'
                        )

                try:
                    if control_mode == 'velocity':
                        if tgt_twist is not None:
                            send_setvelocity(ctrl_sock,
                                             tgt_twist[0], tgt_twist[1], tgt_twist[2])
                        else:
                            send_setvelocity(ctrl_sock, 0.0, 0.0, 0.0)
                    elif control_mode == 'twist':
                        if tgt_twist is not None:
                            vx_b, vy_b = _world_to_body(
                                tgt_twist[0], tgt_twist[1], auv_yaw)
                            send_settwist(ctrl_sock,
                                          vx_b, vy_b, tgt_twist[2],
                                          0.0, tgt_twist[3], 0.0)
                        else:
                            send_settwist(ctrl_sock, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    else:
                        send_setpose(ctrl_sock, cmd_x, cmd_y, cmd_z, yaw_to_tgt)
                except OSError as e:
                    print(f'[NAV] 控制连接断开：{e} → FAILED')
                    state = FAILED
                    break

            # 调试可视化
            vis = draw_debug(
                static_vis, views.L, l_thresh,
                waypoints, cursor,
                auv_x, auv_y, auv_yaw,
                goal_x, goal_y,
                transformer, state, replan_total, replan_streak,
                last_local_reason, combined_safe, last_map_stats)
            cv2.imshow('Navigate', vis)
            if _vwriter is not None:
                _vwriter.write(vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 节流到 10 Hz
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        print(f'[NAV] 结束  state={state}  replans={replan_total}  streak={replan_streak}')
        if _vwriter is not None:
            _vwriter.release()
            print(f'[NAV] 录像已保存 → {_video_path}')
        pose_listener.stop()
        frame_worker.stop()
        ctrl_sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
