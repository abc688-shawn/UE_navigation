#!/usr/bin/env python3
"""
depth_test_route.py
沿预设直线路线 (-1000,0,200) → (500,0,200) 匀速行驶，
逐帧记录「真实剩余距离（到墙 x=500）」vs「SGBM 中央区域测得深度」，
用于诊断双目感知在不同距离下的精度与有效率。

用法：
    python scripts/depth_test_route.py

输出：
  - cv2 窗口：左目（叠加测量信息）+ 深度伪彩图
  - result/depth_test_log.csv：每帧日志（可用 Excel/pandas 分析）
"""

import csv
import json
import math
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 路线参数 ──────────────────────────────────────────────────────────────────
START    = (-1000.0, 0.0, 200.0)   # 起点 (x, y, z) cm
GOAL     = (  500.0, 0.0, 200.0)   # 终点 cm（墙所在位置）
WALL_X   =   500.0                 # 墙的 X 坐标 cm
YAW_DEG  =   0.0                   # 全程朝 +X，yaw=0
STEP_CM  =   50.0                  # waypoint 间距 cm
REACH_CM =   40.0                  # 判定到达 waypoint 的门限 cm（缩小使终点更精确）
CTRL_HZ  =   10                    # 控制频率
WP_TIMEOUT_S = 20.0                # 单个 waypoint 超时（跳过）

# ── 相机参数（与 UE 及 depthV2.py 一致）──────────────────────────────────────
IMG_W      = 1920
IMG_H      = 1080
FOV_DEG    = 90.0
BASELINE_M = 0.03
FOCAL_PX   = (IMG_W / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)   # ≈ 960 px

# ── SGBM 参数（与 depthV2.py 默认一致）──────────────────────────────────────
NUM_DISP   = 128
BLOCK_SIZE = 5
MIN_DISP   = 0
UNIQUENESS = 15
SPECKLE_W  = 100
SPECKLE_R  = 1

# ── 路径 ─────────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(__file__)
_CFG      = os.path.join(_DIR, '..', 'config', 'navigation_config.json')
LOG_PATH    = os.path.join(_DIR, '..', 'result', 'depth_test_log.csv')
VID_PATH    = os.path.join(_DIR, '..', 'result', 'depth_test_video.mp4')
RECORD_VIDEO = False   # True = 录制视频到 VID_PATH；False = 仅显示窗口

# ── 中文字体（PIL 渲染，cv2.putText 不支持 CJK）────────────────────────────
_FONT_CANDIDATES = [
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
]
_font_cache: dict = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    _font_cache[size] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_texts(img: np.ndarray,
              entries: list,   # [(text, (x,y), color_bgr), ...]
              font_size: int = 26) -> np.ndarray:
    """用 PIL 在 cv2 图像上批量渲染中文文本，返回新 BGR 图像。"""
    font = _get_font(font_size)
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, pos, bgr in entries:
        rgb = (bgr[2], bgr[1], bgr[0])
        draw.text(pos, text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── TCP 工具 ──────────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("TCP 连接断开")
        buf += chunk
    return buf


def _recv_frame(sock: socket.socket):
    ls = struct.unpack("!I", _recv_exact(sock, 4))[0]
    lj = _recv_exact(sock, ls)
    rs = struct.unpack("!I", _recv_exact(sock, 4))[0]
    rj = _recv_exact(sock, rs)
    return lj, rj


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _send_pose(sock: socket.socket, x, y, z, yaw):
    sock.sendall(f'SetPose:{x:.1f},{y:.1f},{z:.1f},0,{yaw:.2f},0\n'.encode())


# ── 位姿监听线程 ──────────────────────────────────────────────────────────────

class _PoseThread(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self._host, self._port = host, port
        self._pose = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                s = socket.socket()
                s.settimeout(2.0)
                s.connect((self._host, self._port))
                buf = b''
                while not self._stop.is_set():
                    try:
                        chunk = s.recv(512)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        self._parse(line.decode(errors='ignore').strip())
                s.close()
            except OSError:
                pass
            time.sleep(0.5)

    def _parse(self, line: str):
        if not line.startswith('Pose:'):
            return
        try:
            v = [float(x) for x in line[5:].split(',')]
            with self._lock:
                self._pose = (v[0], v[1], v[2], v[4])
        except Exception:
            pass

    def get(self):
        with self._lock:
            return self._pose

    def stop(self):
        self._stop.set()


# ── 深度计算 ──────────────────────────────────────────────────────────────────

def _build_sgbm() -> cv2.StereoSGBM:
    P1 = 8  * BLOCK_SIZE ** 2
    P2 = 32 * BLOCK_SIZE ** 2
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=UNIQUENESS,
        speckleWindowSize=SPECKLE_W,
        speckleRange=SPECKLE_R,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


_clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def compute_depth(left_bgr: np.ndarray, right_bgr: np.ndarray,
                  sgbm: cv2.StereoSGBM) -> np.ndarray:
    lg = _clahe_obj.apply(cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY))
    rg = _clahe_obj.apply(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY))
    # 垂直基线：旋转使极线变为水平，满足 SGBM 约定
    lr = cv2.rotate(lg, cv2.ROTATE_90_CLOCKWISE)
    rr = cv2.rotate(rg, cv2.ROTATE_90_CLOCKWISE)
    dr = sgbm.compute(lr, rr)
    d  = cv2.rotate(dr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # CV_16S 视差 → 深度（米）
    valid = d > 0
    depth = np.zeros(d.shape, np.float32)
    df    = np.zeros_like(depth)
    df[valid] = d[valid].astype(np.float32) / 16.0
    depth[valid] = (FOCAL_PX * BASELINE_M) / df[valid]
    return depth


def forward_depth(depth_map: np.ndarray):
    """
    提取图像中央区域（行 40-60%，列 45-55%）的有效深度中位数。
    返回 (median_m, valid_ratio)；无有效点时返回 (0.0, 0.0)。
    """
    h, w = depth_map.shape
    region = depth_map[int(h * 0.40):int(h * 0.60),
                       int(w * 0.45):int(w * 0.55)]
    valid = region[region > 0]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.median(valid)), valid.size / region.size


def depth_colormap(depth_map: np.ndarray, max_m: float = 15.0) -> np.ndarray:
    valid = depth_map > 0
    norm  = np.zeros(depth_map.shape, np.uint8)
    norm[valid] = ((1.0 - np.clip(depth_map[valid], 0, max_m) / max_m) * 255
                   ).astype(np.uint8)
    c = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    c[~valid] = 0
    return c


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    cfg = json.load(open(_CFG))['ue_tcp']

    # 生成沿 X 轴的等距 waypoints
    total = GOAL[0] - START[0]        # 1500 cm
    n     = int(round(total / STEP_CM))
    waypoints = [(START[0] + i / n * total, START[1], START[2])
                 for i in range(n + 1)]
    print(f"[test] 路线: {START} → {GOAL}  共 {len(waypoints)} 个 waypoint")
    print(f"[test] 提示：双目基线={BASELINE_M}m，可靠测距范围约 1.5~2m，"
          f"鱼到墙 <{int(FOCAL_PX*BASELINE_M*100)} cm 时深度才稳定")

    # 连接控制端口
    ctrl = socket.socket()
    ctrl.settimeout(5.0)
    ctrl.connect((cfg['control_host'], cfg['control_port']))
    ctrl.settimeout(None)
    print(f"[test] 控制端口 ✓ {cfg['control_host']}:{cfg['control_port']}")

    # 位姿线程
    pose_th = _PoseThread(cfg['pose_host'], cfg['pose_port'])
    pose_th.start()

    # 监听帧端口
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg['frame_host'], cfg['frame_port']))
    srv.listen(1)
    print(f"[test] 等待 UE 帧连接 {cfg['frame_host']}:{cfg['frame_port']} ...")
    conn, addr = srv.accept()
    print(f"[test] 帧连接来自 {addr}")

    sgbm = _build_sgbm()

    # 日志文件
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_f = open(LOG_PATH, 'w', newline='')
    writer = csv.writer(log_f)
    writer.writerow(['time_s', 'x_cm', 'true_dist_m', 'meas_depth_m',
                     'valid_ratio', 'error_m', 'in_range'])

    # 先把鱼传送到起点
    print(f"[test] 发送鱼到起点 {START}...")
    _send_pose(ctrl, *START, YAW_DEG)
    t0 = time.time()
    while time.time() - t0 < 30.0:
        p = pose_th.get()
        if p and abs(p[0] - START[0]) < 200 and abs(p[1] - START[1]) < 200:
            break
        time.sleep(0.2)
    p = pose_th.get()
    print(f"[test] 鱼当前位置: {p}  开始测试行驶...")

    # CV 窗口
    WIN = "深度测距测试"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 480)

    # 视频录制（由 RECORD_VIDEO 控制）
    VID_W, VID_H = 1280, 480
    if RECORD_VIDEO:
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        vwriter = cv2.VideoWriter(VID_PATH, fourcc, 15.0, (VID_W, VID_H))
        print(f"[test] 视频录制 → {VID_PATH}")
    else:
        vwriter = None
        print("[test] 视频录制已关闭（RECORD_VIDEO=False）")

    cursor     = 0
    t_start    = time.time()
    t_ctrl     = 0.0
    t_wp_enter = time.time()

    try:
        while cursor < len(waypoints):
            # 接收双目帧
            try:
                lj, rj = _recv_frame(conn)
            except ConnectionError as e:
                print(f"[test] 帧断开: {e}")
                break
            left  = _decode(lj)
            right = _decode(rj)
            if left is None or right is None:
                continue

            now = time.time() - t_start

            # SGBM 深度
            depth     = compute_depth(left, right, sgbm)
            meas, vr  = forward_depth(depth)

            # 当前位姿
            pose = pose_th.get()
            cx   = pose[0] if pose else START[0]
            cy   = pose[1] if pose else START[1]

            # 真实距离（cm → m）
            true_dist = max(0.0, (WALL_X - cx) / 100.0)
            # 双目可靠范围：f*B = 960*0.03 ≈ 28.8 px·m；可靠深度 < ~2m
            in_range  = true_dist < 2.0

            # 误差
            if meas > 0:
                error = meas - true_dist
                err_s = f"{error:+.3f}"
            else:
                error = float('nan')
                err_s = "N/A"

            writer.writerow([f'{now:.2f}', f'{cx:.1f}', f'{true_dist:.3f}',
                              f'{meas:.3f}', f'{vr:.3f}', err_s,
                              '1' if in_range else '0'])

            # 控制：每 1/CTRL_HZ 秒发一次
            if time.time() - t_ctrl >= 1.0 / CTRL_HZ:
                t_ctrl = time.time()
                tgt = waypoints[cursor]
                _send_pose(ctrl, tgt[0], tgt[1], tgt[2], YAW_DEG)

                dist_wp = math.hypot(cx - tgt[0], cy - tgt[1])
                timed_out = (time.time() - t_wp_enter) > WP_TIMEOUT_S

                if dist_wp < REACH_CM or timed_out:
                    if timed_out:
                        print(f"[test] waypoint {cursor} 超时，强制跳过")
                    cursor += 1
                    t_wp_enter = time.time()
                    if cursor < len(waypoints):
                        mark = " ◀ 进入测距范围" if in_range else ""
                        print(f"[test] → wp {cursor:2d}/{len(waypoints)}  "
                              f"x={cx:.0f}cm  "
                              f"真实={true_dist:.2f}m  "
                              f"测量={meas:.2f}m  "
                              f"有效={vr*100:.0f}%"
                              f"{mark}")

            # ── 可视化 ────────────────────────────────────────────────────────
            dcolor = depth_colormap(depth, max_m=15.0)

            # 深度图：画中央测量框 + 中文标签
            rh, rw = depth.shape
            r0, r1 = int(rh * 0.40), int(rh * 0.60)
            c0, c1 = int(rw * 0.45), int(rw * 0.55)
            box_color = (0, 255, 0) if meas > 0 else (0, 80, 255)
            cv2.rectangle(dcolor, (c0, r0), (c1, r1), box_color, 3)
            dcolor = put_texts(dcolor,
                               [("测量区", (c0, max(0, r0 - 32)), box_color)],
                               font_size=28)

            # 左目：进入测距范围时加绿框
            left_show = left.copy()
            if in_range:
                cv2.rectangle(left_show, (0, 0),
                              (left_show.shape[1], left_show.shape[0]),
                              (0, 255, 0), 8)

            # 左目：中文文字叠加（PIL 渲染）
            err_color = ((0, 255, 0)   if meas > 0 and abs(error) < 0.5
                         else (0, 100, 255) if meas > 0
                         else (150, 150, 150))
            depth_line = (f"SGBM 测量深度: {meas:.3f} m" if meas > 0
                          else "SGBM 测量深度: 无有效视差")
            range_line = "◀ 已进入测距范围 ▶" if in_range else "超出可靠范围（需距墙 <200 cm）"
            range_clr  = (0, 220, 80) if in_range else (80, 80, 255)

            text_entries = [
                (f"位置: ({cx:.0f}, {cy:.0f}) cm",       (20, 20),  (0, 255, 255)),
                (f"Waypoint: {cursor}/{len(waypoints)}", (20, 56),  (200, 200, 200)),
                (f"真实剩余距离: {true_dist:.3f} m",      (20, 110), (0, 200, 255)),
                (depth_line,                              (20, 148), (0, 255, 150) if meas > 0
                                                                      else (0, 80, 200)),
                (f"有效像素率: {vr*100:.0f}%",            (20, 186), (200, 200, 200)),
                (f"误差: {err_s} m",                      (20, 224), err_color),
                (range_line,                              (20, 278), range_clr),
            ]
            left_show = put_texts(left_show, text_entries, font_size=28)

            # 拼合显示帧
            lh2, lw2 = 480, 640
            left_rsz  = cv2.resize(left_show, (lw2, lh2))
            depth_rsz = cv2.resize(dcolor,    (lw2, lh2))
            disp = np.hstack([left_rsz, depth_rsz])

            cv2.imshow(WIN, disp)
            if vwriter is not None:
                vwriter.write(disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[test] 用户中止")
                break

        print(f"\n[test] 完成。日志 → {LOG_PATH}")
        print(f"[test] 视频 → {VID_PATH}")

    finally:
        conn.close()
        srv.close()
        ctrl.close()
        pose_th.stop()
        log_f.close()
        if vwriter is not None:
            vwriter.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
