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

import copy

import numpy as np
import cv2

# ── 把 scripts 目录加到 sys.path ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from depthV2 import (
    recv_stereo_frame, decode_jpeg, apply_clahe,
    build_stereo_sgbm, apply_wls_filter,
    disparity_to_depth,
    FOCAL_PX, IMAGE_W, IMAGE_H,
    DEFAULT_NUM_DISP, DEFAULT_BLOCK_SIZE, DEFAULT_MIN_DISP,
    DEFAULT_UNIQUENESS, DEFAULT_SPECKLE_W, DEFAULT_SPECKLE_R,
    DEFAULT_WLS_SIGMA, DEFAULT_WLS_LAMBDA,
    HAS_WLS,
)
from astar_planner_v2 import (
    load_config,
    ConfigurationSpace, CoordinateTransformer,
    HybridAStarPlanner, HybridAStarConfig,
    GradientDescentSmoother, ha_path_to_pixels,
    resample_path, compute_yaw, smooth_path_rdp,
)

# ── 文件路径 ──────────────────────────────────────────────────────
_BASE        = os.path.join(_SCRIPT_DIR, '..')
NAV_CFG_PATH  = os.path.join(_BASE, 'config', 'navigation_config.json')
GRID_CFG_PATH = os.path.join(_BASE, 'config', 'grid_config.json')

# ── 状态常量 ──────────────────────────────────────────────────────
IDLE, NAV, REACHED_GOAL, FAILED = 'IDLE', 'NAV', 'REACHED_GOAL', 'FAILED'

# ── 录像开关（True=录制 mp4，False=跳过）─────────────────────────
RECORD_VIDEO = False


def normalize_angle_deg(angle: float) -> float:
    """归一化角度到 [-180, 180)。"""
    return (angle + 180.0) % 360.0 - 180.0


def angular_diff_deg(a: float, b: float) -> float:
    return normalize_angle_deg(a - b)


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
    def __init__(self, listen_host: str, port: int, timeout_s: float):
        super().__init__(daemon=True)
        self._host    = listen_host
        self._port    = port
        self._timeout = timeout_s
        self._q       = queue.Queue(maxsize=1)
        self._stop    = threading.Event()

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
                print(f'[Frame] UE 已连接：{addr}')
                conn.settimeout(self._timeout)
                self._recv_loop(conn)
                conn.close()
            except OSError as e:
                print(f'[Frame] 套接字错误：{e}')
                time.sleep(1.0)

    def _recv_loop(self, conn):
        lmbda_val      = DEFAULT_WLS_LAMBDA * 100    # 8000
        sigma_val      = DEFAULT_WLS_SIGMA  / 10.0   # 1.5
        use_wls        = HAS_WLS and DEFAULT_WLS_SIGMA > 0

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

            # 垂直基线旋转技巧（见 file/双目相机水平转垂直基线改动说明.md）
            left_rot  = cv2.rotate(left_gray,  cv2.ROTATE_90_CLOCKWISE)
            right_rot = cv2.rotate(right_gray, cv2.ROTATE_90_CLOCKWISE)
            disp_rot  = self._stereo.compute(left_rot, right_rot)
            if use_wls:
                disp_rot = apply_wls_filter(
                    self._stereo, left_rot, right_rot, disp_rot,
                    lmbda=lmbda_val, sigma_color=sigma_val)
            disp  = cv2.rotate(disp_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth = disparity_to_depth(disp)  # float32 H×W，米，0=无效

            # 非阻塞入队，丢弃旧帧，保留最新
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(depth)

    def get_depth(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════
# 深度 → 世界坐标端点投影
# ══════════════════════════════════════════════════════════════════
def project_depth_to_world(depth: np.ndarray,
                           auv_x: float, auv_y: float,
                           nav_yaw_deg: float,
                           cfg_depth: dict):
    """
    返回 (obs_endpoints, free_endpoints)。
    obs_endpoints : 障碍命中点的世界坐标列表，供 log-odds 标障碍 + 清自由段使用。
    free_endpoints: 视野内无障碍方向延伸至 r_max 的端点列表，整段射线均标自由格。
    """
    H, W   = depth.shape
    # 开运算过滤孤立深度噪点：仅清除孤立有效像素，保持 0=无效 不变
    _kernel = np.ones((3, 3), np.uint8)
    _valid  = (depth > 0).astype(np.uint8)
    _valid  = cv2.morphologyEx(_valid, cv2.MORPH_OPEN, _kernel)
    depth   = depth * _valid.astype(np.float32)

    r_min  = cfg_depth['min_range_m']
    r_max  = cfg_depth['max_range_m']
    v_lo   = int(cfg_depth['row_band_frac'][0] * H)
    v_hi   = int(cfg_depth['row_band_frac'][1] * H)
    c_band = cfg_depth.get('col_band_frac', [0.0, 1.0])
    c_lo   = int(c_band[0] * W)
    c_hi   = int(c_band[1] * W)
    stride      = cfg_depth['col_stride']
    free_stride = max(stride * 4, 32)   # 自由射线用更稀疏采样（每方向一条即可）
    cx     = IMAGE_W / 2.0
    f      = FOCAL_PX
    yaw_r  = math.radians(nav_yaw_deg)

    obs_endpoints  = []
    free_endpoints = []
    col_slice = depth[v_lo:v_hi, :]   # shape: (row_band, W)
    min_hits = int(cfg_depth.get('min_valid_per_col', 4))
    hit_pct = float(cfg_depth.get('hit_percentile', 20.0))

    # ── 障碍端点：每列取近侧百分位，避免单个噪声像素直接写入地图 ──
    for u in range(c_lo, c_hi, stride):
        col = col_slice[:, u]
        valid = col[(col > r_min) & (col < r_max)]
        if valid.size < min_hits:
            continue
        r = float(np.percentile(valid, hit_pct))
        theta = math.atan2(u - cx, f)
        phi   = yaw_r + theta
        obs_endpoints.append((auv_x + r * 100.0 * math.cos(phi),
                               auv_y + r * 100.0 * math.sin(phi)))

    # ── 自由端点：默认不把“无深度”当作自由空间，避免纹理少/反光导致误清障碍 ──
    # 若仿真场景深度稳定，可在配置里打开 clear_unknown_as_free 恢复旧行为。
    if cfg_depth.get('clear_unknown_as_free', False):
        for u in range(c_lo, c_hi, free_stride):
            col = col_slice[:, u]
            has_obs = np.any((col > r_min) & (col < r_max))
            if not has_obs:
                theta = math.atan2(u - cx, f)
                phi   = yaw_r + theta
                free_endpoints.append((auv_x + r_max * 100.0 * math.cos(phi),
                                        auv_y + r_max * 100.0 * math.sin(phi)))

    return obs_endpoints, free_endpoints


# ══════════════════════════════════════════════════════════════════
# Bresenham 直线算法
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


# ══════════════════════════════════════════════════════════════════
# Log-odds 占用更新
# ══════════════════════════════════════════════════════════════════
def update_occupancy(L: np.ndarray,
                     auv_u: int, auv_v: int,
                     obs_endpoints_cm,
                     transformer: CoordinateTransformer,
                     cfg_log: dict,
                     free_endpoints_cm=()):
    l_free = cfg_log['l_free']
    l_occ  = cfg_log['l_occ']
    l_min  = cfg_log['l_min']
    l_max  = cfg_log['l_max']
    H, W   = L.shape

    # ── 障碍射线：沿途标自由，端点标障碍 ──
    for (xe, ye) in obs_endpoints_cm:
        eu, ev = transformer.world_to_pixel(xe, ye)
        ray = _bresenham(auv_u, auv_v, eu, ev)
        for (ru, rv) in ray[:-1]:
            if 0 <= rv < H and 0 <= ru < W:
                L[rv, ru] = np.clip(L[rv, ru] + l_free, l_min, l_max)
        if 0 <= ev < H and 0 <= eu < W:
            L[ev, eu] = np.clip(L[ev, eu] + l_occ, l_min, l_max)

    # ── 自由射线：整段（含端点）均标自由，清除面向开阔水体方向的残留障碍标记 ──
    for (xe, ye) in free_endpoints_cm:
        eu, ev = transformer.world_to_pixel(xe, ye)
        for (ru, rv) in _bresenham(auv_u, auv_v, eu, ev):
            if 0 <= rv < H and 0 <= ru < W:
                L[rv, ru] = np.clip(L[rv, ru] + l_free, l_min, l_max)


def build_navigation_maps(static_np: np.ndarray,
                          L: np.ndarray,
                          l_thresh: float,
                          inflation_px: int):
    """
    返回 (combined_raw, combined_safe)。
    combined_raw  : 原始占用判定，用于交给规划器做配置空间膨胀。
    combined_safe : 动态障碍已经膨胀，用于在线路径验证和局部避障。
    """
    occ = L > l_thresh
    combined_raw = static_np & ~occ
    safe_r = max(1, int(math.ceil(inflation_px / 2)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * safe_r + 1, 2 * safe_r + 1))
    occ_safe = cv2.dilate(occ.astype(np.uint8), kernel) > 0
    combined_safe = static_np & ~occ_safe
    return combined_raw, combined_safe


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
    返回 front/left/right 的保守距离估计（米），以及是否命中硬避障距离。
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
    sectors = {}
    for name, (a, b) in slices.items():
        vals = band[:, a:b]
        valid = vals[(vals > r_min) & (vals < r_max)]
        if valid.size < min_valid:
            sectors[name] = {
                'distance_m': float('inf'),
                'valid_px': int(valid.size),
                'seen': False,
            }
        else:
            sectors[name] = {
                'distance_m': float(np.percentile(valid, pct)),
                'valid_px': int(valid.size),
                'seen': True,
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

    step = min(max(min_step, dist_to_wp), local_step)
    direct_x = auv_x + step * math.cos(math.radians(goal_yaw))
    direct_y = auv_y + step * math.sin(math.radians(goal_yaw))
    direct_uv = transformer.world_to_pixel(direct_x, direct_y)
    direct_ok = segment_is_free(auv_uv, direct_uv, combined_safe, skip_start_px=2)

    front_m = depth_scan['front']['distance_m'] if depth_scan else float('inf')
    must_avoid = front_m < hard_m
    should_avoid = must_avoid or front_m < caution_m or not direct_ok

    if not should_avoid:
        return direct_x, direct_y, z_fixed, goal_yaw, 'track'

    left_m = depth_scan['left']['distance_m'] if depth_scan else float('inf')
    right_m = depth_scan['right']['distance_m'] if depth_scan else float('inf')
    prefer_sign = -1.0 if left_m >= right_m else 1.0
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

    # 地图候选都被堵住时，最后尝试短距离后退，避免继续顶障碍。
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


def postprocess_hybrid_path(raw_path,
                            cspace: ConfigurationSpace,
                            transformer: CoordinateTransformer,
                            ha_cfg: HybridAStarConfig,
                            z_fixed: float,
                            interval_cm: float):
    """
    Hybrid A* 在线路径后处理：
    节点路径 → RDP 简化 → 梯度下降平滑 → 重采样。
    """
    path_pixels = ha_path_to_pixels(raw_path, transformer)
    path_simplified = smooth_path_rdp(path_pixels, epsilon=2.0)

    path_m = []
    for u, v in path_simplified:
        x_cm, y_cm, _ = transformer.pixel_to_world(u, v)
        path_m.append((x_cm / 100.0, y_cm / 100.0))

    if len(path_m) >= 3 and ha_cfg.smooth_iterations > 0:
        smoother = GradientDescentSmoother(path_m, cspace, transformer, ha_cfg)
        path_m = smoother.smooth()

    world_cm = [(x * 100.0, y * 100.0, z_fixed) for x, y in path_m]
    return resample_path(world_cm, interval_cm=interval_cm)


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

    # 起点周围强制保通：外围动态障碍膨胀后可能封堵起点，inflate 前先清通
    keepout_px = int(math.ceil(
        grid_config['planning']['inflation_radius_m']
        * 100.0 / grid_config['resolution']['cm_per_pixel'])) + 1
    su, sv = transformer.world_to_pixel(auv_x, auv_y)
    combined[max(0, sv - keepout_px):min(H, sv + keepout_px + 1),
             max(0, su - keepout_px):min(W, su + keepout_px + 1)] = True

    # 终点周围强制保通：防止 log-odds 噪声把终点标为障碍导致规划失败
    gu, gv = transformer.world_to_pixel(goal_x, goal_y)
    combined[max(0, gv - keepout_px):min(H, gv + keepout_px + 1),
             max(0, gu - keepout_px):min(W, gu + keepout_px + 1)] = True

    grid_list = combined.tolist()

    cspace = ConfigurationSpace(grid_list, grid_config)
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
               state: str, replan_count: int,
               local_reason: str = 'track') -> np.ndarray:
    base = (static_vis * 200).astype(np.uint8)
    vis  = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # 红色 = 动态障碍
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
        f'State: {state}   Replans: {replan_count}',
        f'AUV ({auv_x:.0f}, {auv_y:.0f})  yaw={nav_yaw_deg:.1f}',
        f'dist_goal={dist:.0f}cm  WP {cursor}/{len(waypoints) if waypoints else 0}',
        f'Local: {local_reason}',
    ]
    for i, txt in enumerate(hud):
        cv2.putText(vis, txt, (4, 14 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return vis


# ══════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════
def main():
    # ── 加载配置 ────────────────────────────────────────────────────
    with open(NAV_CFG_PATH)  as f: nav_cfg  = json.load(f)
    grid_config = load_config(GRID_CFG_PATH)

    sp        = nav_cfg['start_pose']
    gp        = nav_cfg['goal_pose']
    z_fixed   = sp['z']
    goal_x,   goal_y = gp['x'], gp['y']

    cfg_depth  = nav_cfg['depth']
    cfg_log    = nav_cfg['log_odds']
    cfg_plan   = nav_cfg['planner']
    cfg_timing = nav_cfg['timing']
    ue_tcp     = nav_cfg['ue_tcp']

    rate_hz       = cfg_timing['control_rate_hz']
    dt            = 1.0 / rate_hz
    pose_init_to  = cfg_timing.get('pose_init_timeout_s', 30.0)
    pose_to       = cfg_timing['pose_timeout_s']
    reconnect_s   = cfg_timing['pose_reconnect_s']
    frame_to      = cfg_timing['frame_timeout_s']
    l_thresh    = cfg_log['l_thresh']

    # ── 初始化全空闲地图（未知环境，仅靠双目实时感知障碍）──────────
    img_cfg = grid_config['image_size']
    H, W   = img_cfg['height'], img_cfg['width']
    static_np  = np.ones((H, W), dtype=bool)   # 全部可通行
    static_vis = static_np.astype(np.float32)  # 可视化用（全白）

    transformer = CoordinateTransformer(grid_config)
    L           = np.zeros((H, W), dtype=np.float32)
    # 预计算膨胀+安全半径（像素），replan 清除和 AUV 轨迹释放共用
    inflation_px = int(math.ceil(
        (grid_config['planning']['inflation_radius_m']
         + grid_config['planning'].get('safety_margin_m', 0.0))
        * 100.0 / grid_config['resolution']['cm_per_pixel'])) * 2

    # ── 启动后台线程 ─────────────────────────────────────────────────
    pose_listener = PoseListener(
        ue_tcp['pose_host'], ue_tcp['pose_port'], reconnect_s)
    frame_worker  = FrameWorker(
        ue_tcp['frame_host'], ue_tcp['frame_port'], frame_to)
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
    replan_count = 0
    last_replan  = 0.0
    last_pose_ok = None   # 首次收到 pose 的时间
    # 进度监控
    min_dist_seen      = float('inf')
    last_progress_time = None
    stall_timeout_s    = cfg_plan.get('stall_timeout_s', 15.0)
    last_depth_scan    = None
    last_depth_time    = 0.0
    last_local_reason  = 'track'

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
                print(f'[NAV] 首帧 pose: x={auv_x:.0f} y={auv_y:.0f} yaw={auv_yaw:.1f}')

                wp = run_replan_with_fallback(
                    auv_x, auv_y, auv_yaw,
                    goal_x, goal_y, z_fixed,
                    static_np, grid_config, transformer, nav_cfg)
                if wp is None:
                    print('[NAV] 初始规划失败 → FAILED')
                    state = FAILED
                    continue
                waypoints   = wp
                cursor      = 0
                last_pose_ok = time.time()
                print(f'[NAV] 初始路径 {len(waypoints)} 个 waypoint → NAV')
                state = NAV
                continue

            # ──────────────────────── NAV ─────────────────────────
            pose  = pose_listener.get_pose()
            depth = frame_worker.get_depth()

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

            # log-odds 时间衰减（旧观测自然消退，半衰期 ≈ 2.2 s @ 10 Hz）
            L *= 0.97
            # AUV 实际经过的位置强制释放（物理已证伪）
            clear_r = max(2, inflation_px // 2)
            _cv0 = max(0, auv_v - clear_r); _cv1 = min(H, auv_v + clear_r + 1)
            _cu0 = max(0, auv_u - clear_r); _cu1 = min(W, auv_u + clear_r + 1)
            L[_cv0:_cv1, _cu0:_cu1] = np.minimum(L[_cv0:_cv1, _cu0:_cu1], 0.0)

            # 深度投影 + log-odds 更新
            if depth is not None:
                last_depth_scan = analyze_depth_scan(depth, cfg_depth)
                last_depth_time = now
                obs_ep, free_ep = project_depth_to_world(
                    depth, auv_x, auv_y, auv_yaw, cfg_depth)
                update_occupancy(L, auv_u, auv_v, obs_ep, transformer, cfg_log, free_ep)
            depth_scan = last_depth_scan if (now - last_depth_time) < 1.0 else None

            # 合并地图（True = 可通行）
            combined, combined_safe = build_navigation_maps(
                static_np, L, l_thresh, inflation_px)

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
            elif (now - last_progress_time) > stall_timeout_s:
                print(f'[NAV] 连续 {stall_timeout_s:.0f}s 无进展 (dist={dist_goal:.0f}cm) → 寻找逃脱路径')
                escape = find_escape_target(
                    auv_x, auv_y, goal_x, goal_y,
                    combined_safe, transformer,
                    cfg_plan.get('escape_dist_cm', 500.0))
                if escape is not None:
                    print(f'[NAV] 逃脱目标 → ({escape[0]:.0f}, {escape[1]:.0f})')
                    # 逃脱点插到路径最前：鱼先探索新区域，再沿后续路径继续
                    tail = waypoints[cursor:] if cursor < len(waypoints) else []
                    waypoints = [(escape[0], escape[1], z_fixed)] + tail
                    cursor = 0
                min_dist_seen = dist_goal
                last_progress_time = now
                last_replan = 0.0

            # waypoint 推进
            while cursor < len(waypoints):
                wp = waypoints[cursor]
                if math.hypot(auv_x - wp[0], auv_y - wp[1]) < cfg_plan['waypoint_reach_cm']:
                    cursor += 1
                else:
                    break
            cursor = min(cursor, len(waypoints) - 1) if waypoints else 0

            # 路径有效性 + 重规划
            need_replan = not validate_plan(
                waypoints, cursor, combined_safe,
                cfg_plan['lookahead_waypoints'], transformer,
                auv_pix=(auv_u, auv_v))
            if need_replan and (now - last_replan) > cfg_plan['replan_cooldown_s']:
                replan_count += 1
                print(f'[NAV] REPLAN #{replan_count}  dist={dist_goal:.0f}cm')
                if replan_count > cfg_plan['max_replans']:
                    print('[NAV] 重规划次数超限 → FAILED')
                    state = FAILED
                    break
                # 清除 AUV 周围 log-odds（至少 5 格，突破局部假障碍陷阱）
                _replan_r = max(5, inflation_px)
                cu, cv = auv_u, auv_v
                v0 = max(0, cv - _replan_r)
                v1 = min(H, cv + _replan_r + 1)
                u0 = max(0, cu - _replan_r)
                u1 = min(W, cu + _replan_r + 1)
                L[v0:v1, u0:u1] = 0.0
                combined, combined_safe = build_navigation_maps(
                    static_np, L, l_thresh, inflation_px)

                new_wp = run_replan_with_fallback(
                    auv_x, auv_y, auv_yaw,
                    goal_x, goal_y, z_fixed,
                    combined, grid_config, transformer, nav_cfg)
                if new_wp is None:
                    print('[NAV] 重规划无解 → FAILED')
                    state = FAILED
                    break
                waypoints   = new_wp
                cursor      = 0
                last_replan = now

            # 发送指令：全局 waypoint 前再过一层近场局部避障，避免直线顶到未知障碍
            if cursor < len(waypoints):
                target = select_tracking_waypoint(
                    auv_x, auv_y, auv_z, waypoints, cursor, cfg_plan)
                if target is None:
                    print('[NAV] 跟踪目标为空 → FAILED')
                    state = FAILED
                    break
                cmd_x, cmd_y, cmd_z, yaw_to_tgt, last_local_reason = choose_control_target(
                    auv_x, auv_y, auv_z, auv_yaw,
                    target, z_fixed, depth_scan,
                    combined_safe, transformer, cfg_plan)
                try:
                    send_setpose(ctrl_sock, cmd_x, cmd_y, cmd_z, yaw_to_tgt)
                except OSError as e:
                    print(f'[NAV] 控制连接断开：{e} → FAILED')
                    state = FAILED
                    break

            # 调试可视化
            vis = draw_debug(
                static_vis, L, l_thresh,
                waypoints, cursor,
                auv_x, auv_y, auv_yaw,
                goal_x, goal_y,
                transformer, state, replan_count,
                last_local_reason)
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
        print(f'[NAV] 结束  state={state}  replans={replan_count}')
        if _vwriter is not None:
            _vwriter.release()
            print(f'[NAV] 录像已保存 → {_video_path}')
        pose_listener.stop()
        frame_worker.stop()
        ctrl_sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
