#!/usr/bin/env python3
"""
stereo_bm.py
接收 UE 双目摄像机画面，计算视差图 → 深度图，支持鼠标测距。

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
"""

import math
import socket
import struct
import sys
import cv2
import numpy as np

# ─── TCP 配置 ────────────────────────────────────────────────────────────────
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8989          # UE FrameStreamingComponent 的 SendPort

# ─── 相机参数（与 UE 中设置一致） ─────────────────────────────────────────────
IMAGE_W     = 1920          # 图像宽度（像素）
IMAGE_H     = 1080          # 图像高度（像素）
FOV_H_DEG   = 90.0          # 水平视场角（度）— 与 UE Camera 的 FieldOfView 一致
BASELINE_M  = 0.03          # 双目基线距离（米）— 两个 Camera 之间的水平间距

# 由 FOV 和分辨率计算像素焦距
_fov_h_rad = math.radians(FOV_H_DEG)
FOCAL_PX   = (IMAGE_W / 2.0) / math.tan(_fov_h_rad / 2.0)  # ≈ 960 px (FOV=90°)

# ─── 默认 BM 参数 ─────────────────────────────────────────────────────────────
DEFAULT_NUM_DISP   = 64     # 视差范围，必须是 16 的倍数
DEFAULT_BLOCK_SIZE = 15     # 匹配块大小，必须是奇数（5~51）
DEFAULT_MIN_DISP   = 0      # 最小视差
DEFAULT_UNIQUENESS = 10     # 唯一性比率（0~100）
DEFAULT_SPECKLE_W  = 100    # 噪点剔除窗口（0 = 关闭）
DEFAULT_SPECKLE_R  = 32     # 噪点剔除范围
DEFAULT_MAX_DEPTH  = 20     # 深度图可视化量程上限（米）

# 是否对输入图像做 CLAHE 预增强（改善低对比度场景下的匹配效果）
USE_CLAHE = True


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


def build_stereo_bm(num_disp, block_size, min_disp, uniqueness,
                    speckle_win, speckle_range) -> cv2.StereoBM:
    """根据参数创建 StereoBM 对象。"""
    # blockSize 必须为奇数且 >= 5
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)

    # numDisparities 必须是 16 的倍数且 > 0
    if num_disp <= 0:
        num_disp = 16
    num_disp = (num_disp // 16) * 16

    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    bm.setMinDisparity(min_disp)
    bm.setUniquenessRatio(uniqueness)
    bm.setSpeckleWindowSize(speckle_win)
    bm.setSpeckleRange(speckle_range)
    bm.setDisp12MaxDiff(1)
    return bm


def disparity_to_color(disp: np.ndarray) -> np.ndarray:
    """
    将 StereoBM 输出的 16 倍定点视差图转为 COLORMAP_JET 彩色图。
    无效点（值 = -16）显示为黑色。
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
    """StereoBM 的 CV_16S 输出 → 浮点视差（像素），同时返回有效掩码。"""
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
    """创建 BM 参数调节窗口。"""
    win = "BM 参数调节"
    cv2.namedWindow(win)
    cv2.createTrackbar("numDisparities x16", win, DEFAULT_NUM_DISP // 16, 16,  _noop)
    cv2.createTrackbar("blockSize (odd)",    win, DEFAULT_BLOCK_SIZE,      51,  _noop)
    cv2.createTrackbar("minDisparity",       win, DEFAULT_MIN_DISP,        32,  _noop)
    cv2.createTrackbar("uniqueness",         win, DEFAULT_UNIQUENESS,      100, _noop)
    cv2.createTrackbar("speckleWin",         win, DEFAULT_SPECKLE_W,       200, _noop)
    cv2.createTrackbar("speckleRange",       win, DEFAULT_SPECKLE_R,       100, _noop)
    cv2.createTrackbar("maxDepth (m)",       win, DEFAULT_MAX_DEPTH,       100, _noop)
    return win


def read_params(win: str) -> dict:
    """从 trackbar 窗口读取当前 BM 参数。"""
    nd = max(1, cv2.getTrackbarPos("numDisparities x16", win)) * 16
    bs = cv2.getTrackbarPos("blockSize (odd)", win)
    if bs % 2 == 0:
        bs += 1
    bs = max(5, bs)
    max_d = max(1, cv2.getTrackbarPos("maxDepth (m)", win))
    return {
        "num_disp":    nd,
        "block_size":  bs,
        "min_disp":    cv2.getTrackbarPos("minDisparity",  win),
        "uniqueness":  cv2.getTrackbarPos("uniqueness",    win),
        "speckle_win": cv2.getTrackbarPos("speckleWin",    win),
        "speckle_rng": cv2.getTrackbarPos("speckleRange",  win),
        "max_depth":   max_d,
    }


# ─── 鼠标测距回调 ─────────────────────────────────────────────────────────────

class DepthProbe:
    """在指定窗口上通过鼠标点击读取深度值。"""

    def __init__(self):
        self.depth_map: np.ndarray | None = None
        self.left_bgr: np.ndarray | None = None
        self.click_pos: tuple[int, int] | None = None
        self.click_depth: float = 0.0

    def update(self, depth_map: np.ndarray, left_bgr: np.ndarray):
        self.depth_map = depth_map
        self.left_bgr = left_bgr

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.depth_map is None:
            return
        h, w = self.depth_map.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            d = float(self.depth_map[y, x])
            self.click_pos = (x, y)
            self.click_depth = d
            if d > 0:
                print(f"[测距] 像素({x}, {y})  深度 = {d:.3f} m")
            else:
                print(f"[测距] 像素({x}, {y})  深度无效（视差不足）")

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """在图像上绘制测距十字线和深度数值。"""
        if self.click_pos is None:
            return img
        out = img.copy()
        cx, cy = self.click_pos
        color = (0, 255, 0)
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
        if self.click_depth > 0:
            label = f"{self.click_depth:.2f} m"
        else:
            label = "N/A"
        cv2.putText(out, label, (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print(f"[stereo_bm] 相机参数: FOV={FOV_H_DEG}°  "
          f"焦距={FOCAL_PX:.1f}px  基线={BASELINE_M}m  "
          f"分辨率={IMAGE_W}x{IMAGE_H}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(1)
    print(f"[stereo_bm] 等待 UE 连接 {LISTEN_HOST}:{LISTEN_PORT} ...")

    conn, addr = server.accept()
    print(f"[stereo_bm] 已连接：{addr}")

    param_win = create_param_window()

    cv2.namedWindow("左目",     cv2.WINDOW_NORMAL)
    cv2.namedWindow("右目",     cv2.WINDOW_NORMAL)
    cv2.namedWindow("视差图",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("深度图",   cv2.WINDOW_NORMAL)

    probe = DepthProbe()
    cv2.setMouseCallback("深度图", probe.on_mouse)
    cv2.setMouseCallback("左目",   probe.on_mouse)

    prev_params = {}
    stereo = build_stereo_bm(
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

            # ── 读取参数，如有变化则重建 BM ──
            params = read_params(param_win)
            bm_params = {k: v for k, v in params.items() if k != "max_depth"}
            if bm_params != {k: v for k, v in prev_params.items() if k != "max_depth"}:
                stereo = build_stereo_bm(
                    params["num_disp"],  params["block_size"],
                    params["min_disp"],  params["uniqueness"],
                    params["speckle_win"], params["speckle_rng"]
                )
            prev_params = params.copy()

            # ── 计算视差 → 深度 ──
            disparity = stereo.compute(left_gray, right_gray)   # CV_16S
            disp_color = disparity_to_color(disparity)
            depth_map  = disparity_to_depth(disparity)
            depth_color = depth_to_color(depth_map, params["max_depth"])

            probe.update(depth_map, left_bgr)

            # ── 叠加信息 ──
            info_disp = (f"numDisp={params['num_disp']}  "
                         f"block={params['block_size']}  "
                         f"minDisp={params['min_disp']}")
            cv2.putText(disp_color, info_disp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            valid_depths = depth_map[depth_map > 0]
            if valid_depths.size > 0:
                info_depth = (f"range: {valid_depths.min():.2f} ~ "
                              f"{valid_depths.max():.2f} m  "
                              f"max_vis={params['max_depth']}m")
            else:
                info_depth = "no valid depth"
            cv2.putText(depth_color, info_depth, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── 绘制测距标记 ──
            left_display  = probe.draw_overlay(left_bgr)
            depth_display = probe.draw_overlay(depth_color)

            # ── 显示 ──
            cv2.imshow("左目",   left_display)
            cv2.imshow("右目",   right_bgr)
            cv2.imshow("视差图", disp_color)
            cv2.imshow("深度图", depth_display)

            key = cv2.waitKey(1) & 0xFF
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

    finally:
        conn.close()
        server.close()
        cv2.destroyAllWindows()
        print("[stereo_bm] 已退出")


if __name__ == "__main__":
    main()
