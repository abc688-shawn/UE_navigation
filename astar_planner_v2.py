"""
Hybrid A* core used by `navigate.py`.

This module keeps only the planning logic that is on the online replanning path:
configuration space, coordinate transforms, Hybrid A* search, optional smoothing,
and collision-aware postprocessing.
"""

import json
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Pixel = Tuple[int, int]
WorldPointCm = Tuple[float, float, float]
WorldPointM = Tuple[float, float]


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def _mod2pi(theta: float) -> float:
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ConfigurationSpace:
    def __init__(self, grid: List[List[bool]], config: dict):
        self.original_grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

        self.cm_per_pixel = config["resolution"]["cm_per_pixel"]
        planning = config.get("planning", {})

        total_inflation_m = (
            planning.get("inflation_radius_m", 1.0)
            + planning.get("safety_margin_m", 0.5)
        )
        self.inflation_pixels = int(
            math.ceil(total_inflation_m * 100.0 / self.cm_per_pixel)
        )
        self.danger_pixels = int(
            math.ceil(planning.get("danger_zone_m", 3.0) * 100.0 / self.cm_per_pixel)
        )
        self.distance_cost_weight = planning.get("distance_cost_weight", 0.3)

        self.inflated_grid: Optional[List[List[bool]]] = None
        self.distance_field: Optional[List[List[float]]] = None

    @staticmethod
    def _disk_offsets(radius: int) -> List[Tuple[int, int]]:
        offsets = []
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                if dv * dv + du * du <= radius * radius:
                    offsets.append((dv, du))
        return offsets

    def inflate_obstacles(self) -> List[List[bool]]:
        print(
            f"  膨胀半径: {self.inflation_pixels} 像素 "
            f"({self.inflation_pixels * self.cm_per_pixel / 100:.2f} m)"
        )

        self.inflated_grid = [[True for _ in range(self.width)] for _ in range(self.height)]
        offsets = self._disk_offsets(self.inflation_pixels)
        for v in range(self.height):
            for u in range(self.width):
                if self.original_grid[v][u]:
                    continue
                for dv, du in offsets:
                    nv, nu = v + dv, u + du
                    if 0 <= nv < self.height and 0 <= nu < self.width:
                        self.inflated_grid[nv][nu] = False

        original_obstacles = sum(
            1
            for v in range(self.height)
            for u in range(self.width)
            if not self.original_grid[v][u]
        )
        inflated_obstacles = sum(
            1
            for v in range(self.height)
            for u in range(self.width)
            if not self.inflated_grid[v][u]
        )

        print(f"  原始障碍格子: {original_obstacles}")
        print(f"  膨胀后障碍格子: {inflated_obstacles}")
        return self.inflated_grid

    def compute_distance_field(self) -> List[List[float]]:
        print(
            f"  危险区域半径: {self.danger_pixels} 像素 "
            f"({self.danger_pixels * self.cm_per_pixel / 100:.2f} m)"
        )

        inf = float("inf")
        self.distance_field = [[inf for _ in range(self.width)] for _ in range(self.height)]

        pq: List[Tuple[float, int, int]] = []
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:
                    self.distance_field[v][u] = 0.0
                    heapq.heappush(pq, (0.0, v, u))

        directions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        while pq:
            dist, v, u = heapq.heappop(pq)
            if dist > self.distance_field[v][u]:
                continue
            if dist > self.danger_pixels:
                continue
            for dv, du, cost in directions:
                nv, nu = v + dv, u + du
                if not (0 <= nv < self.height and 0 <= nu < self.width):
                    continue
                new_dist = dist + cost
                if new_dist >= self.distance_field[nv][nu]:
                    continue
                self.distance_field[nv][nu] = new_dist
                heapq.heappush(pq, (new_dist, nv, nu))

        return self.distance_field

    def get_danger_penalty(self, v: int, u: int) -> float:
        if self.distance_field is None:
            return 0.0
        dist = self.distance_field[v][u]
        if dist >= self.danger_pixels:
            return 0.0
        return 1.0 - dist / self.danger_pixels


class CoordinateTransformer:
    def __init__(self, config: dict):
        bounds = config["world_bounds"]
        self.x_min = bounds["x_min"]
        self.x_max = bounds["x_max"]
        self.y_min = bounds["y_min"]
        self.y_max = bounds["y_max"]
        self.z_fixed = bounds["z_fixed"]

        self.cm_per_pixel = config["resolution"]["cm_per_pixel"]

        img_size = config["image_size"]
        self.width = img_size["width"]
        self.height = img_size["height"]

        coord_conv = config["coordinate_convention"]
        self.flip_v = coord_conv.get("flip_v", False)

    def pixel_to_world(self, u: int, v: int) -> WorldPointCm:
        if self.flip_v:
            v = self.height - 1 - v
        x = self.x_min + (u + 0.5) * self.cm_per_pixel
        y = self.y_min + (v + 0.5) * self.cm_per_pixel
        return (x, y, self.z_fixed)

    def world_to_pixel(self, x: float, y: float) -> Pixel:
        u = int((x - self.x_min) / self.cm_per_pixel)
        v = int((y - self.y_min) / self.cm_per_pixel)
        if self.flip_v:
            v = self.height - 1 - v
        u = max(0, min(u, self.width - 1))
        v = max(0, min(v, self.height - 1))
        return (u, v)

def resample_path(
    world_path: List[WorldPointCm], interval_cm: float = 200.0
) -> List[WorldPointCm]:
    """
    Split each existing path segment so that every output segment is at most
    `interval_cm` long. This preserves turns instead of reconnecting sparse
    waypoints with corner-cutting straight lines.
    """
    if len(world_path) < 2 or interval_cm <= 0:
        return list(world_path)

    resampled = [tuple(map(float, world_path[0]))]
    for start, end in zip(world_path, world_path[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        steps = max(1, int(math.ceil(seg_len / interval_cm)))
        for step in range(1, steps + 1):
            t = step / steps
            point = (
                start[0] + dx * t,
                start[1] + dy * t,
                start[2] + dz * t,
            )
            if math.hypot(point[0] - resampled[-1][0], point[1] - resampled[-1][1]) > 1e-9:
                resampled.append(point)
    if resampled[-1] != tuple(map(float, world_path[-1])):
        resampled.append(tuple(map(float, world_path[-1])))
    return resampled


def compute_yaw(p1: WorldPointCm, p2: WorldPointCm) -> float:
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


@dataclass
class HybridAStarConfig:
    r_min_m: float = 2.0
    L_base_m: float = 1.0
    gamma_step: float = 0.5
    d0_step_m: float = 5.0
    L_min_factor: float = 0.4
    L_max_factor: float = 1.5
    n_steerings: int = 5
    allow_reverse: bool = False

    n_theta_bins: int = 36

    lambda_danger: float = 0.3
    w_heading: float = 0.5
    w_reversal: float = 2.0
    heading_penalty_threshold: float = math.pi / 4

    goal_pos_tol_m: float = 1.0
    max_iterations: int = 500_000

    w_smooth: float = 0.3
    w_obstacle: float = 0.5
    w_curvature: float = 0.2
    d_safe_m: float = 1.5
    kappa_max: float = 0.5
    smooth_iterations: int = 200
    armijo_alpha: float = 0.3
    armijo_beta: float = 0.5
    armijo_init_step: float = 0.1

    @classmethod
    def from_config(cls, config: dict) -> "HybridAStarConfig":
        data = config.get("hybrid_astar", {})
        obj = cls()
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, type(getattr(obj, key))(value))
        if "kappa_max" not in data:
            obj.kappa_max = 1.0 / obj.r_min_m
        return obj


class HybridAStarNode:
    __slots__ = ("x", "y", "theta", "g", "h", "parent", "is_forward")

    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        g: float = 0.0,
        h: float = 0.0,
        parent: Optional["HybridAStarNode"] = None,
        is_forward: bool = True,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.parent = parent
        self.is_forward = is_forward

    @property
    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, other: "HybridAStarNode") -> bool:
        return self.f < other.f


class DubinsPath:
    @staticmethod
    def compute_min_length(
        x1: float,
        y1: float,
        t1: float,
        x2: float,
        y2: float,
        t2: float,
        radius: float,
    ) -> float:
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            d_theta = abs(_normalize_angle(t2 - t1))
            return radius * min(d_theta, 2.0 * math.pi - d_theta)

        d = distance / radius
        angle = math.atan2(dy, dx)
        alpha = _mod2pi(t1 - angle)
        beta = _mod2pi(t2 - angle)

        lengths = []
        for word in ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL"):
            length = DubinsPath._word_length(d, alpha, beta, word)
            if length is not None and length >= 0:
                lengths.append(length)
        if not lengths:
            return distance
        return radius * min(lengths)

    @staticmethod
    def _word_length(d: float, alpha: float, beta: float, word: str) -> Optional[float]:
        fn = {
            "LSL": DubinsPath._LSL,
            "RSR": DubinsPath._RSR,
            "LSR": DubinsPath._LSR,
            "RSL": DubinsPath._RSL,
            "RLR": DubinsPath._RLR,
            "LRL": DubinsPath._LRL,
        }[word]
        return fn(d, alpha, beta)

    @staticmethod
    def _LSL(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (sa - sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(cb - ca, d + sa - sb)
        t = _mod2pi(-alpha + tmp)
        q = _mod2pi(beta - tmp)
        return t + p + q

    @staticmethod
    def _RSR(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (sb - sa)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca - cb, d - sa + sb)
        t = _mod2pi(alpha - tmp)
        q = _mod2pi(-beta + tmp)
        return t + p + q

    @staticmethod
    def _LSR(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        p_sq = -2 + d * d + 2 * math.cos(alpha - beta) + 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        t = _mod2pi(-alpha + tmp)
        q = _mod2pi(-beta + tmp)
        return t + p + q

    @staticmethod
    def _RSL(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        p_sq = -2 + d * d + 2 * math.cos(alpha - beta) - 2 * d * (sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        t = _mod2pi(alpha - tmp)
        q = _mod2pi(beta - tmp)
        return t + p + q

    @staticmethod
    def _RLR(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        tmp = (6.0 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (sa - sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2.0 * math.pi - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
        q = _mod2pi(alpha - beta - t + _mod2pi(p))
        return t + p + q

    @staticmethod
    def _LRL(d: float, alpha: float, beta: float) -> Optional[float]:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta), math.cos(beta)
        tmp = (6.0 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-sa + sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2.0 * math.pi - math.acos(tmp))
        t = _mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
        return t + p + q


class HybridAStarPlanner:
    def __init__(
        self,
        cspace: ConfigurationSpace,
        cfg: HybridAStarConfig,
        transformer: CoordinateTransformer,
    ):
        self.cspace = cspace
        self.cfg = cfg
        self.transformer = transformer
        self.pixel_size_m = cspace.cm_per_pixel / 100.0

    def _meters_to_pixel(self, x_m: float, y_m: float) -> Pixel:
        return self.transformer.world_to_pixel(x_m * 100.0, y_m * 100.0)

    def _is_valid_pixel(self, u: int, v: int) -> bool:
        if 0 <= v < self.cspace.height and 0 <= u < self.cspace.width:
            return self.cspace.inflated_grid[v][u]
        return False

    def _dynamic_step_length(self, dist_to_goal_m: float) -> float:
        cfg = self.cfg
        gain = 0.5 * (1.0 + math.tanh(cfg.gamma_step * (dist_to_goal_m - cfg.d0_step_m)))
        scale = cfg.L_min_factor + (cfg.L_max_factor - cfg.L_min_factor) * gain
        return cfg.L_base_m * scale

    def _get_curvatures(self) -> List[float]:
        kmax = 1.0 / self.cfg.r_min_m
        if self.cfg.n_steerings <= 1:
            return [0.0]
        return [
            -kmax + (2.0 * kmax * i) / (self.cfg.n_steerings - 1)
            for i in range(self.cfg.n_steerings)
        ]

    def _apply_motion(
        self, x: float, y: float, theta: float, arc_len: float, curvature: float
    ) -> Tuple[float, float, float]:
        delta_theta = curvature * arc_len
        if abs(delta_theta) < 1e-6:
            return (
                x + arc_len * math.cos(theta),
                y + arc_len * math.sin(theta),
                theta,
            )

        radius = arc_len / delta_theta
        x_new = x + radius * (math.sin(theta + delta_theta) - math.sin(theta))
        y_new = y - radius * (math.cos(theta + delta_theta) - math.cos(theta))
        return x_new, y_new, _normalize_angle(theta + delta_theta)

    def _arc_collision_free(
        self, x0: float, y0: float, theta0: float, arc_len: float, curvature: float
    ) -> bool:
        checks = max(3, int(abs(arc_len) / (0.5 * self.pixel_size_m)))
        for i in range(1, checks + 1):
            ratio = i / checks
            x_i, y_i, _ = self._apply_motion(x0, y0, theta0, arc_len * ratio, curvature)
            u, v = self._meters_to_pixel(x_i, y_i)
            if not self._is_valid_pixel(u, v):
                return False
        return True

    def _discretize_state(self, x_m: float, y_m: float, theta: float) -> Tuple[int, int, int]:
        u, v = self._meters_to_pixel(x_m, y_m)
        theta_norm = (_normalize_angle(theta) + math.pi) / (2.0 * math.pi)
        theta_bin = int(theta_norm * self.cfg.n_theta_bins) % self.cfg.n_theta_bins
        return (u, v, theta_bin)

    def _heuristic_dubins(
        self, x: float, y: float, theta: float, gx: float, gy: float, g_theta: float
    ) -> float:
        return DubinsPath.compute_min_length(x, y, theta, gx, gy, g_theta, self.cfg.r_min_m)

    def search(
        self,
        start_pixel: Pixel,
        goal_pixel: Pixel,
        start_theta: Optional[float] = None,
    ) -> Optional[List[HybridAStarNode]]:
        su, sv = start_pixel
        gu, gv = goal_pixel

        sx_cm, sy_cm, _ = self.transformer.pixel_to_world(su, sv)
        gx_cm, gy_cm, _ = self.transformer.pixel_to_world(gu, gv)
        sx, sy = sx_cm / 100.0, sy_cm / 100.0
        gx, gy = gx_cm / 100.0, gy_cm / 100.0

        if start_theta is None:
            start_theta = math.atan2(gy - sy, gx - sx)

        if not self._is_valid_pixel(su, sv):
            if not (
                0 <= sv < self.cspace.height
                and 0 <= su < self.cspace.width
                and self.cspace.original_grid[sv][su]
            ):
                print(f"  错误：起点 ({su}, {sv}) 不可通行")
                return None
            print("  警告：起点在膨胀区域内，继续搜索...")

        if not self._is_valid_pixel(gu, gv):
            if not (
                0 <= gv < self.cspace.height
                and 0 <= gu < self.cspace.width
                and self.cspace.original_grid[gv][gu]
            ):
                print(f"  错误：终点 ({gu}, {gv}) 不可通行")
                return None
            print("  警告：终点在膨胀区域内，继续搜索...")

        goal_theta = math.atan2(gy - sy, gx - sx)
        h0 = self._heuristic_dubins(sx, sy, start_theta, gx, gy, goal_theta)
        start_node = HybridAStarNode(sx, sy, start_theta, g=0.0, h=h0)

        open_heap: List[Tuple[float, int, HybridAStarNode]] = []
        heapq.heappush(open_heap, (start_node.f, 0, start_node))

        g_best: Dict[Tuple[int, int, int], float] = {
            self._discretize_state(sx, sy, start_theta): 0.0
        }
        closed_set = set()
        curvatures = self._get_curvatures()
        counter = 0
        iterations = 0

        while open_heap and iterations < self.cfg.max_iterations:
            iterations += 1
            _, _, node = heapq.heappop(open_heap)

            if math.hypot(node.x - gx, node.y - gy) <= self.cfg.goal_pos_tol_m:
                path = self._reconstruct_path(node)
                print(f"  Hybrid A* 完成：迭代 {iterations} 次，路径 {len(path)} 节点")
                return path

            state_key = self._discretize_state(node.x, node.y, node.theta)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            step_len = self._dynamic_step_length(math.hypot(gx - node.x, gy - node.y))
            directions = [True]
            if self.cfg.allow_reverse:
                directions.append(False)

            for is_forward in directions:
                signed_step = step_len if is_forward else -step_len
                for curvature in curvatures:
                    xn, yn, tn = self._apply_motion(
                        node.x, node.y, node.theta, signed_step, curvature
                    )
                    un, vn = self._meters_to_pixel(xn, yn)
                    if not self._is_valid_pixel(un, vn):
                        continue
                    if not self._arc_collision_free(
                        node.x, node.y, node.theta, signed_step, curvature
                    ):
                        continue

                    arc_cost = abs(signed_step)
                    danger_cost = (
                        self.cspace.get_danger_penalty(vn, un) * self.cfg.lambda_danger
                    )
                    delta_heading = abs(_normalize_angle(tn - node.theta))
                    heading_cost = 0.0
                    if delta_heading > self.cfg.heading_penalty_threshold:
                        heading_cost = self.cfg.w_heading * (
                            delta_heading - self.cfg.heading_penalty_threshold
                        )
                    reversal_cost = 0.0
                    if node.parent is not None and is_forward != node.is_forward:
                        reversal_cost = self.cfg.w_reversal

                    g_new = node.g + arc_cost + danger_cost + heading_cost + reversal_cost
                    next_key = self._discretize_state(xn, yn, tn)
                    if next_key in closed_set:
                        continue
                    if next_key in g_best and g_best[next_key] <= g_new:
                        continue

                    g_best[next_key] = g_new
                    g_theta = math.atan2(gy - yn, gx - xn)
                    h_new = self._heuristic_dubins(xn, yn, tn, gx, gy, g_theta)

                    child = HybridAStarNode(
                        x=xn,
                        y=yn,
                        theta=tn,
                        g=g_new,
                        h=h_new,
                        parent=node,
                        is_forward=is_forward,
                    )
                    counter += 1
                    heapq.heappush(open_heap, (child.f, counter, child))

        print(f"  Hybrid A* 搜索失败：达到最大迭代 {iterations} 次")
        return None

    @staticmethod
    def _reconstruct_path(node: HybridAStarNode) -> List[HybridAStarNode]:
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


class GradientDescentSmoother:
    def __init__(
        self,
        path_meters: List[WorldPointM],
        cspace: ConfigurationSpace,
        transformer: CoordinateTransformer,
        cfg: HybridAStarConfig,
    ):
        self.path_meters = path_meters
        self.cspace = cspace
        self.transformer = transformer
        self.cfg = cfg
        self.pixel_size_m = cspace.cm_per_pixel / 100.0

    def _meters_to_pixel(self, x_m: float, y_m: float) -> Pixel:
        return self.transformer.world_to_pixel(x_m * 100.0, y_m * 100.0)

    def _dist_at_m(self, x_m: float, y_m: float) -> float:
        u, v = self._meters_to_pixel(x_m, y_m)
        u = max(0, min(u, self.cspace.width - 1))
        v = max(0, min(v, self.cspace.height - 1))
        d_pix = self.cspace.distance_field[v][u]
        if d_pix == float("inf"):
            return self.cfg.d_safe_m * 10.0
        return d_pix * self.pixel_size_m

    def _is_valid_m(self, x_m: float, y_m: float) -> bool:
        u, v = self._meters_to_pixel(x_m, y_m)
        if 0 <= v < self.cspace.height and 0 <= u < self.cspace.width:
            return self.cspace.inflated_grid[v][u]
        return False

    @staticmethod
    def _kappa(X: np.ndarray, i: int) -> float:
        a_vec = X[i] - X[i - 1]
        b_vec = X[i + 1] - X[i]
        la = float(np.linalg.norm(a_vec))
        lb = float(np.linalg.norm(b_vec))
        if la < 1e-10 or lb < 1e-10:
            return 0.0
        cross = float(a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0])
        chord = X[i + 1] - X[i - 1]
        lc = float(np.linalg.norm(chord))
        if lc < 1e-10:
            return 0.0
        return 2.0 * abs(cross) / (la * lb * lc)

    def _F_s(self, X: np.ndarray) -> float:
        cost = 0.0
        for i in range(1, len(X) - 1):
            delta = X[i + 1] - 2 * X[i] + X[i - 1]
            cost += float(np.dot(delta, delta))
        return self.cfg.w_smooth * cost

    def _F_o(self, X: np.ndarray) -> float:
        cost = 0.0
        for i in range(1, len(X) - 1):
            penalty = max(0.0, self.cfg.d_safe_m - self._dist_at_m(X[i, 0], X[i, 1]))
            cost += penalty * penalty
        return self.cfg.w_obstacle * cost

    def _F_r(self, X: np.ndarray) -> float:
        cost = 0.0
        for i in range(1, len(X) - 1):
            excess = max(0.0, self._kappa(X, i) - self.cfg.kappa_max)
            cost += excess * excess
        return self.cfg.w_curvature * cost

    def _total_cost(self, X: np.ndarray) -> float:
        return self._F_s(X) + self._F_o(X) + self._F_r(X)

    def _grad_smoothness(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        grad = np.zeros_like(X)
        for j in range(1, N - 1):
            d_jm1 = (X[j] - 2 * X[j - 1] + X[j - 2]) if j >= 2 else np.zeros(2)
            d_j = X[j + 1] - 2 * X[j] + X[j - 1]
            d_jp1 = (X[j + 2] - 2 * X[j + 1] + X[j]) if j <= N - 3 else np.zeros(2)
            grad[j] = 2.0 * self.cfg.w_smooth * (d_jm1 - 2 * d_j + d_jp1)
        return grad

    def _grad_obstacle_numerical(self, X: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(X)
        eps = self.pixel_size_m
        for i in range(1, len(X) - 1):
            dist = self._dist_at_m(X[i, 0], X[i, 1])
            penalty = max(0.0, self.cfg.d_safe_m - dist)
            if penalty <= 0.0:
                continue
            dx = (
                self._dist_at_m(X[i, 0] + eps, X[i, 1])
                - self._dist_at_m(X[i, 0] - eps, X[i, 1])
            ) / (2.0 * eps)
            dy = (
                self._dist_at_m(X[i, 0], X[i, 1] + eps)
                - self._dist_at_m(X[i, 0], X[i, 1] - eps)
            ) / (2.0 * eps)
            grad[i] = -2.0 * self.cfg.w_obstacle * penalty * np.array([dx, dy])
        return grad

    def _grad_curvature_numerical(self, X: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(X)
        eps = 1e-4
        for i in range(1, len(X) - 1):
            if self._kappa(X, i) <= self.cfg.kappa_max:
                continue
            for axis in range(2):
                Xp = X.copy()
                Xm = X.copy()
                Xp[i, axis] += eps
                Xm[i, axis] -= eps
                grad[i, axis] = (self._F_r(Xp) - self._F_r(Xm)) / (2.0 * eps)
        return grad

    def _armijo_step(self, X: np.ndarray, grad: np.ndarray, current_cost: float) -> float:
        grad_norm_sq = float(np.sum(grad[1:-1] ** 2))
        if grad_norm_sq < 1e-15:
            return self.cfg.armijo_init_step

        alpha = self.cfg.armijo_init_step
        for _ in range(30):
            trial = X.copy()
            trial[1:-1] -= alpha * grad[1:-1]
            if self._total_cost(trial) <= current_cost - self.cfg.armijo_alpha * alpha * grad_norm_sq:
                return alpha
            alpha *= self.cfg.armijo_beta
        return alpha

    def smooth(self) -> List[WorldPointM]:
        if len(self.path_meters) < 3:
            return list(self.path_meters)

        X = np.array(self.path_meters, dtype=float)
        X_orig = X.copy()

        for iteration in range(self.cfg.smooth_iterations):
            cost = self._total_cost(X)
            grad = (
                self._grad_smoothness(X)
                + self._grad_obstacle_numerical(X)
                + self._grad_curvature_numerical(X)
            )
            grad_max = float(np.max(np.abs(grad[1:-1]))) if len(X) > 2 else 0.0
            if grad_max < 1e-7:
                print(f"    梯度下降收敛于第 {iteration} 次迭代")
                break

            alpha = self._armijo_step(X, grad, cost)

            candidate = X.copy()
            candidate[1:-1] -= alpha * grad[1:-1]
            if all(self._is_valid_m(candidate[i, 0], candidate[i, 1]) for i in range(1, len(candidate) - 1)):
                X = candidate
                continue

            fallback = X.copy()
            fallback[1:-1] -= (alpha * 0.1) * grad[1:-1]
            if all(self._is_valid_m(fallback[i, 0], fallback[i, 1]) for i in range(1, len(fallback) - 1)):
                X = fallback

        for i in range(1, len(X) - 1):
            if not self._is_valid_m(X[i, 0], X[i, 1]):
                print("    警告：平滑路径有碰撞，回退至平滑前")
                return [(float(p[0]), float(p[1])) for p in X_orig]

        return [(float(p[0]), float(p[1])) for p in X]

def _bresenham_line(u0: int, v0: int, u1: int, v1: int) -> List[Pixel]:
    points = []
    du = abs(u1 - u0)
    dv = abs(v1 - v0)
    step_u = 1 if u0 < u1 else -1
    step_v = 1 if v0 < v1 else -1
    err = du - dv
    u, v = u0, v0

    while True:
        points.append((u, v))
        if u == u1 and v == v1:
            break
        err2 = 2 * err
        if err2 > -dv:
            err -= dv
            u += step_u
        if err2 < du:
            err += du
            v += step_v
    return points


def _segments_collision_free(
    path_m: List[WorldPointM],
    cspace: ConfigurationSpace,
    transformer: CoordinateTransformer,
) -> bool:
    if len(path_m) < 2:
        return True
    for (x0, y0), (x1, y1) in zip(path_m, path_m[1:]):
        u0, v0 = transformer.world_to_pixel(x0 * 100.0, y0 * 100.0)
        u1, v1 = transformer.world_to_pixel(x1 * 100.0, y1 * 100.0)
        for u, v in _bresenham_line(u0, v0, u1, v1):
            if not (0 <= v < cspace.height and 0 <= u < cspace.width):
                return False
            if not cspace.inflated_grid[v][u]:
                return False
    return True


def postprocess_hybrid_path(
    raw_path: List[HybridAStarNode],
    cspace: ConfigurationSpace,
    transformer: CoordinateTransformer,
    cfg: HybridAStarConfig,
    z_fixed: float,
    interval_cm: float,
) -> List[WorldPointCm]:
    """
    Online path postprocess pipeline:
    raw hybrid path -> optional smoothing -> collision check -> segment subdivision.
    """
    path_m = [(node.x, node.y) for node in raw_path]

    if len(path_m) >= 3 and cfg.smooth_iterations > 0:
        smoothed = GradientDescentSmoother(path_m, cspace, transformer, cfg).smooth()
        if _segments_collision_free(smoothed, cspace, transformer):
            path_m = smoothed
        else:
            print("    警告：平滑后路段穿过障碍，回退到未平滑路径")

    world_cm = [(x * 100.0, y * 100.0, z_fixed) for x, y in path_m]
    return resample_path(world_cm, interval_cm=interval_cm)
