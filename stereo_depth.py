#!/usr/bin/env python3
"""
stereo_depth.py
双目深度感知模块：接收 UE 双目帧流，使用 StereoSGBM + WLS 滤波计算深度图。

库接口（供 navigate.py / occupancy_mapper.py 导入）：
  recv_stereo_frame, decode_jpeg, apply_clahe,
  build_stereo_sgbm, apply_wls_filter, disparity_to_depth,
  disparity_to_float, stereo_confidence_to_float,
  FOCAL_PX, IMAGE_W, IMAGE_H, BASELINE_M, …

独立运行（调试/标定）：
  python stereo_depth.py   —— 打开实时双目视窗，支持鼠标测距与参数调节

TCP 协议（与 FrameStreamingComponent 对应）：
  UE 连接本机 LISTEN_PORT（8989），每帧发送：
    [4B left_size (big-endian)] [left JPEG]
    [4B right_size (big-endian)] [right JPEG]

使用方法：
  1. 先运行本脚本，等待 UE 连接
  2. 启动 UE 仿真
  3. 在「深度图」或「左目」窗口上点击鼠标左键 → 显示该点深度（米）
  4. 按 S 保存当前帧、视差图和深度图
  5. 按 Q 退出

调参说明（trackbar 窗口）：
  - numDisparities : 视差搜索范围（16 的倍数），越大搜索范围越广，速度越慢
  - blockSize      : 匹配块大小（奇数），越大越平滑但细节损失越多
  - minDisparity   : 最小视差起点，相机基线较小时通常为 0
  - uniqueness     : 唯一性比率过滤，越大过滤越严格（去伪匹配）
  - speckleWin     : 噪点剔除窗口大小（0 = 不过滤）
  - speckleRange   : 噪点剔除视差范围
  - maxDepth       : 深度图可视化最大量程（米）
  - wlsSigma x10   : WLS 滤波 sigma_color（0 = 关闭 WLS）
  - wlsLambda x100 : WLS 滤波平滑强度
"""

import json
import math
import os
import socket
import struct
import sys
import threading
import time
import cv2
import numpy as np

# ─── TCP 配置 ────────────────────────────────────────────────────────────────
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8989          # UE FrameStreamingComponent 的 SendPort

# ─── 手动控制配置 ─────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'navigation_config.json')
MOVE_STEP_CM = 100.0        # 每次按键移动距离（cm）
MOVE_STEP_Z_CM = 50.0       # 每次按键升降距离（cm）

# macOS 下 OpenCV waitKey 不带掩码时的方向键值
_KEY_UP    = 63232
_KEY_DOWN  = 63233
_KEY_LEFT  = 63234
_KEY_RIGHT = 63235
# 部分平台/配置下带掩码后的方向键值（备用）
_KEY_UP_M    = 0
_KEY_DOWN_M  = 1
_KEY_LEFT_M  = 2
_KEY_RIGHT_M = 3


def _load_ctrl_cfg() -> dict:
    try:
        with open(_CFG_PATH) as f:
            return json.load(f)['ue_tcp']
    except Exception:
        return {'control_host': '10.106.2.223', 'control_port': 12345,
                'pose_host':    '10.106.2.223', 'pose_port':    8991}


# ─── 位姿监听线程（复用 navigate.py 的协议） ──────────────────────────────────

class _PoseListener(threading.Thread):
    """后台线程：连接 UE PoseTelemetryComponent (:8991)，实时更新位姿。"""

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self._host  = host
        self._port  = port
        self._lock  = threading.Lock()
        self._pose  = None          # (x_cm, y_cm, z_cm, yaw_deg)
        self._stop  = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self._host, self._port))
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
                time.sleep(0.5)

    def _parse(self, line: str):
        # 格式：Pose:x,y,z,pitch,yaw,roll
        if not line.startswith('Pose:'):
            return
        try:
            vals = [float(v) for v in line[5:].split(',')]
            x, y, z, _pitch, yaw, _roll = vals
            with self._lock:
                self._pose = (x, y, z, yaw)
        except Exception:
            pass

    def get_pose(self):
        with self._lock:
            return self._pose

    def stop(self):
        self._stop.set()


def _send_setpose(sock: socket.socket, x: float, y: float,
                  z: float, yaw: float):
    cmd = f'SetPose:{x:.1f},{y:.1f},{z:.1f},0,{yaw:.2f},0\n'
    sock.sendall(cmd.encode())

# ─── 相机参数（从 navigation_config.json ["camera"] 读取，硬编码值作为兜底）──
def _load_camera_cfg() -> dict:
    try:
        with open(_CFG_PATH, encoding='utf-8') as f:
            return json.load(f).get('camera', {})
    except Exception:
        return {}

_cam_cfg   = _load_camera_cfg()
IMAGE_W    = int(_cam_cfg.get('image_w',    1920))
IMAGE_H    = int(_cam_cfg.get('image_h',    1080))
FOV_H_DEG  = float(_cam_cfg.get('fov_h_deg', 90.0))
BASELINE_M = float(_cam_cfg.get('baseline_m', 0.10))

# 由 FOV 和分辨率计算像素焦距（所有下游模块 import 此值，无需各自重复计算）
_fov_h_rad = math.radians(FOV_H_DEG)
FOCAL_PX   = (IMAGE_W / 2.0) / math.tan(_fov_h_rad / 2.0)

# ─── 默认 SGBM 参数 ───────────────────────────────────────────────────────────
DEFAULT_NUM_DISP   = 192    # 视差范围（16 的倍数）；192 对应最近可靠测距约 0.15m
DEFAULT_BLOCK_SIZE = 5      # 匹配块大小（奇数 1~11）；SGBM 用小块即可
DEFAULT_MIN_DISP   = 0      # 最小视差
DEFAULT_UNIQUENESS = 15     # 唯一性比率（0~100）；收紧以过滤重复纹理弱匹配
DEFAULT_SPECKLE_W  = 100    # 噪点剔除窗口（0 = 关闭）
DEFAULT_SPECKLE_R  = 1      # 噪点剔除范围
DEFAULT_MAX_DEPTH  = 20     # 深度图可视化量程上限（米）
DEFAULT_WLS_SIGMA  = 15     # WLS sigma_color × 10（= 1.5），0 = 关闭
DEFAULT_WLS_LAMBDA = 80     # WLS lambda × 100（= 8000）

# 是否对输入图像做 CLAHE 预增强（改善低对比度场景下的匹配效果）
USE_CLAHE = True

# 检测 WLS 滤波是否可用（需要 opencv-contrib-python 提供 ximgproc）
try:
    _wls_test = cv2.ximgproc.createDisparityWLSFilter
    HAS_WLS = True
except AttributeError:
    HAS_WLS = False
    print("[stereo_bm] 警告: cv2.ximgproc 不可用，WLS 滤波已关闭。"
          "安装 opencv-contrib-python 可启用。")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """阻塞接收，直到恰好收到 n 个字节。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("TCP 连接断开")
        buf += chunk
    return buf


def recv_stereo_frame(sock: socket.socket):
    """
    接收一帧双目数据。
    返回 (left_jpeg: bytes, right_jpeg: bytes)。
    若连接断开则抛出 ConnectionError。
    """
    left_size  = struct.unpack("!I", recv_exact(sock, 4))[0]
    left_jpeg  = recv_exact(sock, left_size)
    right_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    right_jpeg = recv_exact(sock, right_size)
    return left_jpeg, right_jpeg


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG 字节流 → BGR uint8 numpy 数组。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE 直方图均衡化，增强局部对比度。"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_stereo_sgbm(num_disp, block_size, min_disp, uniqueness,
                      speckle_win, speckle_range) -> cv2.StereoSGBM:
    """根据参数创建 StereoSGBM 对象。"""
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(1, block_size)

    if num_disp <= 0:
        num_disp = 16
    num_disp = (num_disp // 16) * 16

    # P1/P2 平滑惩罚：控制视差变化的连续性，对低纹理场景至关重要
    cn = 1  # 灰度图通道数
    P1 = 8 * cn * block_size * block_size
    P2 = 32 * cn * block_size * block_size

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_win,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm


def apply_wls_filter(sgbm, left_gray, right_gray, disp_left,
                     lmbda, sigma_color):
    """
    WLS (Weighted Least Squares) 滤波：用左右视差一致性 + 边缘引导填补空洞。
    返回 (filtered_disp_CV_16S, confidence_map_uint8)。
    confidence 来自 wls.getConfidenceMap()：0=最不可信，255=最可信。
    """
    right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
    disp_right = right_matcher.compute(right_gray, left_gray)

    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
    wls.setLambda(lmbda)
    wls.setSigmaColor(sigma_color)
    filtered = wls.filter(disp_left, left_gray,
                          disparity_map_right=disp_right)
    confidence = wls.getConfidenceMap()   # CV_8UC1, 0–255
    return filtered, confidence


def stereo_confidence_to_float(conf_uint8: np.ndarray,
                                valid_mask: np.ndarray) -> np.ndarray:
    """WLS 置信图 uint8[0,255] → float32[0,1]；无效视差像素置 0。"""
    conf = conf_uint8.astype(np.float32) / 255.0
    conf[~valid_mask] = 0.0
    return conf


def disparity_to_color(disp: np.ndarray) -> np.ndarray:
    """
    将 StereoSGBM 输出的 16 倍定点视差图转为 COLORMAP_JET 彩色图。
    无效点显示为黑色。
    """
    disp_float, valid_mask = disparity_to_float(disp)

    disp_norm = np.zeros_like(disp_float, dtype=np.uint8)
    if disp_float[valid_mask].size > 0:
        min_val = disp_float[valid_mask].min()
        max_val = disp_float[valid_mask].max()
        if max_val > min_val:
            disp_norm[valid_mask] = (
                (disp_float[valid_mask] - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)

    color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    color[~valid_mask] = (0, 0, 0)
    return color


def disparity_to_float(disp: np.ndarray):
    """StereoSGBM 的 CV_16S 输出 → 浮点视差（像素），同时返回有效掩码。"""
    valid_mask = disp > 0
    disp_float = np.zeros(disp.shape, dtype=np.float32)
    disp_float[valid_mask] = disp[valid_mask].astype(np.float32) / 16.0
    return disp_float, valid_mask


def disparity_to_depth(disp: np.ndarray) -> np.ndarray:
    """
    视差图 → 深度图（米）。
    公式：Z = f * B / d，其中 f = FOCAL_PX（像素），B = BASELINE_M（米）。
    无效视差点深度设为 0。
    """
    disp_float, valid = disparity_to_float(disp)
    depth = np.zeros_like(disp_float)
    depth[valid] = (FOCAL_PX * BASELINE_M) / disp_float[valid]
    return depth


def depth_to_color(depth: np.ndarray, max_depth_m: float) -> np.ndarray:
    """
    深度图（米）→ 伪彩色可视化。
    近处为暖色（红/黄），远处为冷色（蓝），超量程/无效点为黑色。
    """
    valid = depth > 0
    depth_clipped = np.clip(depth, 0, max_depth_m)

    depth_norm = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        depth_norm[valid] = (
            (1.0 - depth_clipped[valid] / max_depth_m) * 255
        ).astype(np.uint8)

    color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)
    return color


# ─── Trackbar 回调（占位，参数在主循环中读取）─────────────────────────────────

def _noop(_): pass


def create_param_window():
    """创建 SGBM + WLS 参数调节窗口。"""
    win = "SGBM 参数调节"
    cv2.namedWindow(win)
    cv2.createTrackbar("numDisparities x16", win, DEFAULT_NUM_DISP // 16, 16,  _noop)
    cv2.createTrackbar("blockSize (odd)",    win, DEFAULT_BLOCK_SIZE,      11,  _noop)
    cv2.createTrackbar("minDisparity",       win, DEFAULT_MIN_DISP,        32,  _noop)
    cv2.createTrackbar("uniqueness",         win, DEFAULT_UNIQUENESS,      100, _noop)
    cv2.createTrackbar("speckleWin",         win, DEFAULT_SPECKLE_W,       400, _noop)
    cv2.createTrackbar("speckleRange",       win, DEFAULT_SPECKLE_R,       10,  _noop)
    cv2.createTrackbar("maxDepth (m)",       win, DEFAULT_MAX_DEPTH,       100, _noop)
    if HAS_WLS:
        cv2.createTrackbar("wlsSigma x10",   win, DEFAULT_WLS_SIGMA,      50,  _noop)
        cv2.createTrackbar("wlsLambda x100", win, DEFAULT_WLS_LAMBDA,     200, _noop)
    return win


def read_params(win: str) -> dict:
    """从 trackbar 窗口读取当前参数。"""
    nd = max(1, cv2.getTrackbarPos("numDisparities x16", win)) * 16
    bs = cv2.getTrackbarPos("blockSize (odd)", win)
    if bs % 2 == 0:
        bs += 1
    bs = max(1, bs)
    max_d = max(1, cv2.getTrackbarPos("maxDepth (m)", win))

    p = {
        "num_disp":    nd,
        "block_size":  bs,
        "min_disp":    cv2.getTrackbarPos("minDisparity",  win),
        "uniqueness":  cv2.getTrackbarPos("uniqueness",    win),
        "speckle_win": cv2.getTrackbarPos("speckleWin",    win),
        "speckle_rng": cv2.getTrackbarPos("speckleRange",  win),
        "max_depth":   max_d,
        "wls_sigma":   0.0,
        "wls_lambda":  0.0,
    }
    if HAS_WLS:
        sigma_raw = cv2.getTrackbarPos("wlsSigma x10", win)
        lam_raw   = cv2.getTrackbarPos("wlsLambda x100", win)
        p["wls_sigma"]  = sigma_raw / 10.0
        p["wls_lambda"] = lam_raw * 100.0
    return p


# ─── 鼠标测距回调 ─────────────────────────────────────────────────────────────

class DepthProbe:
    """在合成画面上通过鼠标点击读取深度值（支持 2×2 布局）。"""

    def __init__(self):
        self.depth_map: np.ndarray | None = None
        self.cell_h: int = 0
        self.cell_w: int = 0
        self.click_pos: tuple[int, int] | None = None
        self.click_depth: float = 0.0

    def update(self, depth_map: np.ndarray, cell_h: int, cell_w: int):
        self.depth_map = depth_map
        self.cell_h = cell_h
        self.cell_w = cell_w

    def on_mouse(self, event, x, y, flags, param):
        """处理 2×2 布局上的点击，左上=左目，右下=深度图，均可测距。"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.depth_map is None:
            return

        col = x // self.cell_w if self.cell_w > 0 else 0
        row = y // self.cell_h if self.cell_h > 0 else 0
        lx = x % self.cell_w
        ly = y % self.cell_h

        # 只在左目（0,0）和深度图（1,1）区域响应
        if (row, col) not in ((0, 0), (1, 1)):
            return

        dh, dw = self.depth_map.shape[:2]
        src_x = int(lx * dw / self.cell_w)
        src_y = int(ly * dh / self.cell_h)
        if not (0 <= src_x < dw and 0 <= src_y < dh):
            return

        d = float(self.depth_map[src_y, src_x])
        self.click_pos = (src_x, src_y)
        self.click_depth = d
        if d > 0:
            print(f"[测距] 像素({src_x}, {src_y})  深度 = {d:.3f} m")
        else:
            print(f"[测距] 像素({src_x}, {src_y})  深度无效（视差不足）")

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """在图像上绘制测距十字线和深度数值。"""
        if self.click_pos is None:
            return img
        out = img.copy()
        h, w = out.shape[:2]
        oh, ow = self.depth_map.shape[:2]
        cx = int(self.click_pos[0] * w / ow)
        cy = int(self.click_pos[1] * h / oh)
        color = (0, 255, 0)
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
        if self.click_depth > 0:
            label = f"{self.click_depth:.2f} m"
        else:
            label = "N/A"
        cv2.putText(out, label, (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out


def make_grid_2x2(imgs, cell_w, cell_h):
    """将 4 张图拼成 2×2 网格，每张缩放到 (cell_w, cell_h)。"""
    resized = []
    for img in imgs:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized.append(cv2.resize(img, (cell_w, cell_h)))
    top = np.hstack(resized[:2])
    bot = np.hstack(resized[2:])
    return np.vstack([top, bot])


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print(f"[stereo_bm] 相机参数: FOV={FOV_H_DEG}°  "
          f"焦距={FOCAL_PX:.1f}px  基线={BASELINE_M}m  "
          f"分辨率={IMAGE_W}x{IMAGE_H}")

    # ── 手动控制：连接 UE 控制端口 ──
    ue_tcp = _load_ctrl_cfg()
    ctrl_sock = None
    try:
        ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ctrl_sock.settimeout(3.0)
        ctrl_sock.connect((ue_tcp['control_host'], ue_tcp['control_port']))
        ctrl_sock.settimeout(None)
        print(f"[stereo_bm] 控制端口已连接 "
              f"{ue_tcp['control_host']}:{ue_tcp['control_port']}  "
              f"（方向键可移动机器鱼）")
    except Exception as e:
        print(f"[stereo_bm] 控制端口连接失败，方向键不可用: {e}")
        ctrl_sock = None

    pose_listener = _PoseListener(ue_tcp['pose_host'], ue_tcp['pose_port'])
    pose_listener.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(1)
    print(f"[stereo_bm] 等待 UE 连接 {LISTEN_HOST}:{LISTEN_PORT} ...")

    conn, addr = server.accept()
    print(f"[stereo_bm] 已连接：{addr}")

    param_win = create_param_window()

    DISPLAY_WIN = "Stereo Depth"
    cell_w, cell_h = 640, 360
    cv2.namedWindow(DISPLAY_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WIN, cell_w * 2, cell_h * 2)

    probe = DepthProbe()
    cv2.setMouseCallback(DISPLAY_WIN, probe.on_mouse)

    prev_sgbm_keys = {}
    stereo = build_stereo_sgbm(
        DEFAULT_NUM_DISP, DEFAULT_BLOCK_SIZE, DEFAULT_MIN_DISP,
        DEFAULT_UNIQUENESS, DEFAULT_SPECKLE_W, DEFAULT_SPECKLE_R
    )

    save_idx = 0

    try:
        while True:
            # ── 接收双目帧 ──
            try:
                left_jpeg, right_jpeg = recv_stereo_frame(conn)
            except ConnectionError as e:
                print(f"[stereo_bm] 连接断开：{e}")
                break

            left_bgr  = decode_jpeg(left_jpeg)
            right_bgr = decode_jpeg(right_jpeg)
            if left_bgr is None or right_bgr is None:
                print("[stereo_bm] JPEG 解码失败，跳过本帧")
                continue

            # ── 转灰度 + 可选 CLAHE ──
            left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            if USE_CLAHE:
                left_gray  = apply_clahe(left_gray)
                right_gray = apply_clahe(right_gray)

            # ── 读取参数，如有变化则重建 SGBM ──
            params = read_params(param_win)
            sgbm_keys = {k: params[k] for k in
                         ("num_disp", "block_size", "min_disp",
                          "uniqueness", "speckle_win", "speckle_rng")}
            if sgbm_keys != prev_sgbm_keys:
                stereo = build_stereo_sgbm(
                    params["num_disp"],  params["block_size"],
                    params["min_disp"],  params["uniqueness"],
                    params["speckle_win"], params["speckle_rng"]
                )
                prev_sgbm_keys = sgbm_keys.copy()

            # ── 计算视差（+ 可选 WLS 滤波）→ 深度 ──
            disparity = stereo.compute(left_gray, right_gray)   # CV_16S

            use_wls = HAS_WLS and params["wls_sigma"] > 0
            if use_wls:
                disparity, _ = apply_wls_filter(
                    stereo, left_gray, right_gray, disparity,
                    lmbda=params["wls_lambda"],
                    sigma_color=params["wls_sigma"],
                )

            disp_color = disparity_to_color(disparity)
            depth_map  = disparity_to_depth(disparity)
            depth_color = depth_to_color(depth_map, params["max_depth"])

            probe.update(depth_map, cell_h, cell_w)

            # ── 叠加标签和信息 ──
            wls_tag = "WLS ON" if use_wls else "WLS OFF"

            cv2.putText(left_bgr, "Left", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(right_bgr, "Right", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            info_disp = (f"Disparity  numDisp={params['num_disp']}  "
                         f"block={params['block_size']}  {wls_tag}")
            cv2.putText(disp_color, info_disp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            valid_depths = depth_map[depth_map > 0]
            total_px = depth_map.size
            valid_pct = valid_depths.size / total_px * 100
            if valid_depths.size > 0:
                info_depth = (f"Depth  {valid_depths.min():.2f}~"
                              f"{valid_depths.max():.2f}m  "
                              f"valid={valid_pct:.1f}%")
            else:
                info_depth = "Depth  no valid"
            cv2.putText(depth_color, info_depth, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── 绘制测距标记 ──
            left_display  = probe.draw_overlay(left_bgr)
            depth_display = probe.draw_overlay(depth_color)

            # ── 方向键状态叠加 ──
            pose = pose_listener.get_pose()
            if pose:
                px, py, pz, pyaw = pose
                ctrl_label = (f"Pos: ({px:.0f}, {py:.0f}, {pz:.0f}) cm  "
                              f"Yaw: {pyaw:.1f}°")
            else:
                ctrl_label = "Pos: 等待位姿..." if ctrl_sock else "控制: 未连接"
            ctrl_hint = ("↑↓←→ 移动  PgUp/PgDn 升降  S 保存  Q 退出"
                         if ctrl_sock else "S 保存  Q 退出")
            cv2.putText(left_display, ctrl_label, (10, cell_h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(left_display, ctrl_hint,  (10, cell_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ── 2×2 拼图显示 ──
            grid = make_grid_2x2(
                [left_display, right_bgr, disp_color, depth_display],
                cell_w, cell_h,
            )
            cv2.imshow(DISPLAY_WIN, grid)

            raw_key = cv2.waitKey(30)
            key = raw_key & 0xFF

            if key == ord("q"):
                print("[stereo_bm] 用户退出")
                break
            elif key == ord("s"):
                cv2.imwrite(f"left_{save_idx:04d}.png",  left_bgr)
                cv2.imwrite(f"right_{save_idx:04d}.png", right_bgr)

                disp_float, _ = disparity_to_float(disparity)
                np.save(f"disparity_{save_idx:04d}.npy", disp_float)
                cv2.imwrite(f"disparity_{save_idx:04d}.png", disp_color)

                np.save(f"depth_{save_idx:04d}.npy", depth_map)
                cv2.imwrite(f"depth_{save_idx:04d}.png", depth_color)

                print(f"[stereo_bm] 已保存第 {save_idx} 组（含深度图）")
                save_idx += 1

            elif ctrl_sock and raw_key in (_KEY_UP, _KEY_DOWN,
                                           _KEY_LEFT, _KEY_RIGHT,
                                           _KEY_UP_M, _KEY_DOWN_M,
                                           _KEY_LEFT_M, _KEY_RIGHT_M,
                                           ord('u'), ord('d')):
                pose = pose_listener.get_pose()
                if pose is None:
                    print("[manual] 尚未收到位姿，无法移动")
                else:
                    px, py, pz, pyaw = pose
                    yaw_rad = math.radians(pyaw)
                    # 前向单位向量（heading 方向）
                    fwd_x =  math.cos(yaw_rad)
                    fwd_y =  math.sin(yaw_rad)
                    # 右向单位向量（右手坐标系）
                    rgt_x = -math.sin(yaw_rad)
                    rgt_y =  math.cos(yaw_rad)

                    if raw_key in (_KEY_UP, _KEY_UP_M):
                        px += fwd_x * MOVE_STEP_CM
                        py += fwd_y * MOVE_STEP_CM
                        label = "↑ 前进"
                    elif raw_key in (_KEY_DOWN, _KEY_DOWN_M):
                        px -= fwd_x * MOVE_STEP_CM
                        py -= fwd_y * MOVE_STEP_CM
                        label = "↓ 后退"
                    elif raw_key in (_KEY_LEFT, _KEY_LEFT_M):
                        px += rgt_x * MOVE_STEP_CM
                        py += rgt_y * MOVE_STEP_CM
                        label = "← 左移"
                    elif raw_key in (_KEY_RIGHT, _KEY_RIGHT_M):
                        px -= rgt_x * MOVE_STEP_CM
                        py -= rgt_y * MOVE_STEP_CM
                        label = "→ 右移"
                    elif key == ord('u'):
                        pz += MOVE_STEP_Z_CM
                        label = "↑Z 上升"
                    elif key == ord('d'):
                        pz -= MOVE_STEP_Z_CM
                        label = "↓Z 下降"

                    try:
                        _send_setpose(ctrl_sock, px, py, pz, pyaw)
                        print(f"[manual] {label} → "
                              f"SetPose({px:.0f}, {py:.0f}, {pz:.0f})"
                              f"  yaw={pyaw:.1f}°")
                    except Exception as e:
                        print(f"[manual] 发送失败: {e}")
                        ctrl_sock = None

    finally:
        conn.close()
        server.close()
        pose_listener.stop()
        if ctrl_sock:
            ctrl_sock.close()
        cv2.destroyAllWindows()
        print("[stereo_bm] 已退出")


if __name__ == "__main__":
    main()
