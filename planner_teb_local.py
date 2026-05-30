"""
planner_teb_local.py — TEB 风格的在线轨迹形变局部规划层。

每帧对全局 Hybrid A* 路径的前 N 个 waypoint 做轻量级梯度形变，代价项：
  F_smooth   : 二阶差分平滑代价
  F_obstacle : 障碍排斥代价（基于实时 distance transform）
  F_kinematic: 曲率约束代价

起点（index=0）锚定为 AUV 当前位置，终点（index=-1）跟随全局路径弱约束。
输出：
  target_pose  : (x_cm, y_cm, z_cm, yaw_deg)  → SetPose 指令
  target_twist : (vx_cm_s, vy_cm_s, vz_cm_s, omega_deg_s) → SetVelocity 备用
"""

import math

import cv2
import numpy as np

from planner_hybrid_astar import CoordinateTransformer, HybridAStarConfig


def _normalize_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _RealtimeCspace:
    """
    从实时 combined_safe 地图（bool H×W）构建轻量级 distance field。
    只暴露 TEB 梯度函数需要的接口：distance_at_pixel / is_free。
    """

    def __init__(self, combined_safe: np.ndarray, cm_per_pixel: float):
        H, W = combined_safe.shape
        self.height     = H
        self.width      = W
        self.cm_per_pixel = cm_per_pixel

        # cv2.distanceTransform：src=1 为自由格，计算每个自由格到最近障碍格的像素距离。
        dist_f32 = cv2.distanceTransform(
            combined_safe.astype(np.uint8), cv2.DIST_L2, 5)
        self._dist = dist_f32   # shape H×W, float32

    def distance_at_pixel(self, u: int, v: int) -> float:
        u = max(0, min(u, self.width  - 1))
        v = max(0, min(v, self.height - 1))
        return float(self._dist[v, u])

    def is_free_pixel(self, u: int, v: int) -> bool:
        return (0 <= v < self.height and 0 <= u < self.width
                and self._dist[v, u] > 0.0)


class TEBLocalPlanner:
    """
    TEB 局部规划器，替代 select_tracking_waypoint + choose_control_target。

    用法::

        teb = TEBLocalPlanner(cfg_teb, ha_cfg, transformer, cm_per_pixel)
        ...
        target_pose, target_twist = teb.update(
            waypoints, cursor, (auv_x, auv_y, auv_z, auv_yaw),
            combined_safe_local, dt)
        if target_pose is not None:
            send_setpose(sock, *target_pose)
    """

    def __init__(self, cfg_teb: dict, ha_cfg: HybridAStarConfig,
                 transformer: CoordinateTransformer, cm_per_pixel: float):
        self._cfg       = cfg_teb
        self._ha_cfg    = ha_cfg
        self._tf        = transformer
        self._cm_per_px = cm_per_pixel
        self._px_size_m = cm_per_pixel / 100.0

    # ── 坐标转换 ──────────────────────────────────────────────────────

    def _to_pixel(self, x_m: float, y_m: float):
        return self._tf.world_to_pixel(x_m * 100.0, y_m * 100.0)

    def _dist_at_m(self, cspace: _RealtimeCspace, x_m: float, y_m: float) -> float:
        u, v = self._to_pixel(x_m, y_m)
        return cspace.distance_at_pixel(u, v) * self._px_size_m

    def _select_window_waypoints(self, waypoints_global, cursor: int,
                                 horizon: int,
                                 auv_x: float, auv_y: float):
        """
        从全局路径中挑出局部 TEB 控制点。

        Hybrid A* 输出的 raw/smoothed path 往往比控制周期稠密得多，直接把
        十几厘米甚至几厘米一档的 waypoint 全喂给曲率项，会让离散曲率估计
        被“超短线段 + 轻微转角”放大，导致局部梯度把第一个控制点推到 AUV
        身后。这里按最小间距做一次稀疏化，让 TEB 看见的是“控制点”而不是
        “采样点”。
        """
        if horizon <= 0:
            return []

        min_spacing_cm = float(self._cfg.get(
            'min_ctrl_pt_spacing_cm',
            max(self._cfg.get('max_pose_step_cm', 20.0) * 2.0, 50.0)))
        if min_spacing_cm <= 1.0:
            return list(waypoints_global[cursor: cursor + horizon])

        selected = []
        last_x, last_y = auv_x, auv_y
        for wp in waypoints_global[cursor:]:
            if math.hypot(wp[0] - last_x, wp[1] - last_y) < min_spacing_cm:
                continue
            selected.append(wp)
            last_x, last_y = wp[0], wp[1]
            if len(selected) >= horizon:
                break

        # 如果当前位置附近的路径点都太密，至少保留当前 waypoint 作为回退目标。
        if not selected and cursor < len(waypoints_global):
            selected.append(waypoints_global[cursor])
        return selected

    # ── 梯度分量 ──────────────────────────────────────────────────────

    def _grad_smooth(self, X: np.ndarray) -> np.ndarray:
        N, grad = len(X), np.zeros_like(X)
        w = float(self._cfg.get('w_smooth', 0.6))
        for j in range(1, N - 1):
            d_jm1 = (X[j] - 2*X[j-1] + X[j-2]) if j >= 2 else np.zeros(2)
            d_j   =  X[j+1] - 2*X[j] + X[j-1]
            d_jp1 = (X[j+2] - 2*X[j+1] + X[j]) if j <= N-3 else np.zeros(2)
            grad[j] = 2.0 * w * (d_jm1 - 2*d_j + d_jp1)
        return grad

    def _grad_obstacle(self, X: np.ndarray, cspace: _RealtimeCspace) -> np.ndarray:
        grad   = np.zeros_like(X)
        w      = float(self._cfg.get('w_obstacle', 0.9))
        d_safe = float(self._cfg.get('d_safe_m', 1.5))
        eps    = self._px_size_m
        for i in range(1, len(X) - 1):
            dist    = self._dist_at_m(cspace, X[i, 0], X[i, 1])
            penalty = max(0.0, d_safe - dist)
            if penalty <= 0.0:
                continue
            dx = (self._dist_at_m(cspace, X[i,0]+eps, X[i,1])
                - self._dist_at_m(cspace, X[i,0]-eps, X[i,1])) / (2*eps)
            dy = (self._dist_at_m(cspace, X[i,0], X[i,1]+eps)
                - self._dist_at_m(cspace, X[i,0], X[i,1]-eps)) / (2*eps)
            grad[i] = -2.0 * w * penalty * np.array([dx, dy])
        return grad

    @staticmethod
    def _kappa(X: np.ndarray, i: int) -> float:
        a = X[i] - X[i-1]; b = X[i+1] - X[i]
        la = float(np.linalg.norm(a)); lb = float(np.linalg.norm(b))
        if la < 1e-10 or lb < 1e-10:
            return 0.0
        cross = float(a[0]*b[1] - a[1]*b[0])
        chord = X[i+1] - X[i-1]
        lc = float(np.linalg.norm(chord))
        if lc < 1e-10:
            return 0.0
        return 2.0 * abs(cross) / (la * lb * lc)

    def _grad_kinematic(self, X: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(X)
        w    = float(self._cfg.get('w_kinematic', 0.3))
        kmax = float(self._ha_cfg.kappa_max)
        eps  = 1e-4
        for i in range(1, len(X) - 1):
            if self._kappa(X, i) <= kmax:
                continue
            for axis in range(2):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, axis] += eps; Xm[i, axis] -= eps
                cost_p = sum(max(0.0, self._kappa(Xp, j) - kmax)**2
                             for j in range(1, len(Xp)-1))
                cost_m = sum(max(0.0, self._kappa(Xm, j) - kmax)**2
                             for j in range(1, len(Xm)-1))
                grad[i, axis] = w * (cost_p - cost_m) / (2*eps)
        return grad

    # ── 主接口 ────────────────────────────────────────────────────────

    def update(self, waypoints_global, cursor: int, auv_pose: tuple,
               combined_safe: np.ndarray, dt: float):
        """
        每帧调用一次，对全局路径窗口做轻量梯度形变，输出下一步控制目标。

        返回
        ----
        target_pose  : (x_cm, y_cm, z_cm, yaw_deg)  用于 SetPose
        target_twist : (vx_cm_s, vy_cm_s, vz_cm_s, omega_deg_s) 用于 SetVelocity
        两者均为 None 时表示无有效目标（路径耗尽）。
        """
        horizon = int(self._cfg.get('horizon_waypoints', 8))
        auv_x, auv_y, auv_z, auv_yaw = auv_pose

        wps = self._select_window_waypoints(
            waypoints_global, cursor, horizon, auv_x, auv_y)
        if not wps:
            return None, None

        # 控制点（米）：index=0 锚定 = AUV 当前位置，后续来自全局 waypoint
        pts_m = [(auv_x / 100.0, auv_y / 100.0)]
        for wp in wps:
            pts_m.append((wp[0] / 100.0, wp[1] / 100.0))

        if len(pts_m) < 2:
            return None, None

        cspace  = _RealtimeCspace(combined_safe, self._cm_per_px)
        X0      = np.array(pts_m, dtype=float)
        X       = X0.copy()   # (N, 2)
        n_iters = int(self._cfg.get('gd_iterations_per_frame', 3))
        step    = float(self._cfg.get('gd_step_size', 0.04))
        max_deform_cm = float(self._cfg.get(
            'max_deform_cm',
            max(self._cfg.get('max_pose_step_cm', 20.0) * 3.0, 60.0)))

        for _ in range(n_iters):
            grad = (self._grad_smooth(X)
                    + self._grad_obstacle(X, cspace)
                    + self._grad_kinematic(X))
            # 仅移动中间可动点；index=0（AUV 位置）锚定不动
            X[1:-1] -= step * grad[1:-1]
            if max_deform_cm > 0.0 and len(X) > 2:
                max_deform_m = max_deform_cm / 100.0
                movable = X[1:-1]
                movable0 = X0[1:-1]
                delta = movable - movable0
                norms = np.linalg.norm(delta, axis=1)
                mask = norms > max_deform_m
                if np.any(mask):
                    scale = (max_deform_m / norms[mask])[:, None]
                    movable[mask] = movable0[mask] + delta[mask] * scale
                    X[1:-1] = movable

        # 目标点 = 形变后轨迹第一个可动点
        tx_m, ty_m = float(X[1, 0]), float(X[1, 1])
        tx_cm      = tx_m * 100.0
        ty_cm      = ty_m * 100.0
        tz_cm      = float(wps[0][2])

        # 兜底：局部优化不允许把第一目标点拉到全局路径的“后方”。
        nominal_vec = np.array([wps[0][0] - auv_x, wps[0][1] - auv_y], dtype=float)
        target_vec  = np.array([tx_cm - auv_x, ty_cm - auv_y], dtype=float)
        nominal_norm = float(np.linalg.norm(nominal_vec))
        target_norm  = float(np.linalg.norm(target_vec))
        if nominal_norm > 1e-6 and target_norm > 1e-6:
            progress = float(np.dot(target_vec, nominal_vec))
            cos_dev = progress / (nominal_norm * target_norm)
            if progress <= 0.0 or cos_dev < math.cos(math.radians(75.0)):
                tx_cm = float(wps[0][0])
                ty_cm = float(wps[0][1])

        yaw_cmd    = math.degrees(math.atan2(ty_cm - auv_y, tx_cm - auv_x))

        # pose 模式：限制单帧步长上限，防止 InterpSpeed 过高时瞬移
        max_step   = float(self._cfg.get('max_pose_step_cm', 20.0))
        dist_to_tgt = math.hypot(tx_cm - auv_x, ty_cm - auv_y)
        if dist_to_tgt > max_step and max_step > 0:
            scale  = max_step / dist_to_tgt
            tx_cm  = auv_x + (tx_cm - auv_x) * scale
            ty_cm  = auv_y + (ty_cm - auv_y) * scale

        target_pose = (tx_cm, ty_cm, tz_cm, yaw_cmd)

        # velocity / twist 模式：以固定最大速度逼近目标，而非"一帧走完"
        max_speed = float(self._cfg.get('max_speed_cm_s', 150.0))
        max_omega = float(self._cfg.get('max_yaw_rate_deg_s', 60.0))
        dist      = math.hypot(tx_cm - auv_x, ty_cm - auv_y)
        yaw_rad   = math.radians(yaw_cmd)
        if dist > 1.0:
            vx = max_speed * math.cos(yaw_rad)
            vy = max_speed * math.sin(yaw_rad)
        else:
            vx, vy = 0.0, 0.0
        yaw_err = _normalize_angle_deg(yaw_cmd - auv_yaw)
        omega   = max(-max_omega, min(max_omega, yaw_err * 3.0))
        target_twist = (vx, vy, 0.0, omega)

        return target_pose, target_twist
