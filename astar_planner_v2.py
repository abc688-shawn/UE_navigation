"""
Hybrid A* 路径规划算法 v2 - 基于 AUV 运动学约束的改进版

改进自 astar_planner_v1.py
基于：AUV路径规划与路径跟踪 (北京理工大学硕士论文 第4章)

改进内容：
1. 混合 A*：基于运动学模型的弧段扩展，状态 (x, y, θ)（公式 4.2）
2. 自适应步长：基于 tanh 的动态步长调整（公式 4.3）
3. Dubins 曲线启发式：考虑最小转弯半径的最短路径估计（公式 4.4）
4. 增强代价函数：加入航向变化惩罚 p₁ 和方向反转惩罚 p₂（公式 4.5）
5. 梯度下降路径平滑：平滑度 F_s + 障碍约束 F_o + 曲率约束 F_r（公式 4.24）
6. 保留 SafeAStarPlanner 作为回退方案
"""

import json
import heapq
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# 角度工具函数
# ============================================================

def _normalize_angle(theta: float) -> float:
    """归一化角度到 [-π, π]"""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def _mod2pi(theta: float) -> float:
    """归一化角度到 [0, 2π]（Dubins 内部使用）"""
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_occupancy_grid(image_path: str, threshold: int = 128) -> List[List[bool]]:
    """
    读取灰度图并转换为二值占据栅格

    Returns:
        grid[v][u]: True = 可通行, False = 障碍
    """
    img = Image.open(image_path).convert('L')
    width, height = img.size
    pixels = img.load()

    grid = []
    for v in range(height):
        row = []
        for u in range(width):
            row.append(pixels[u, v] >= threshold)
        grid.append(row)

    return grid


# ============================================================
# 配置空间处理（与 v1 相同）
# ============================================================

class ConfigurationSpace:
    """
    配置空间处理器：障碍物膨胀 + 距离场计算
    """

    def __init__(self, grid: List[List[bool]], config: dict):
        self.original_grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

        self.cm_per_pixel = config['resolution']['cm_per_pixel']

        planning = config.get('planning', {})

        inflation_m = planning.get('inflation_radius_m', 1.0)
        safety_m = planning.get('safety_margin_m', 0.5)
        total_inflation_m = inflation_m + safety_m
        self.inflation_pixels = int(math.ceil(total_inflation_m * 100 / self.cm_per_pixel))

        danger_m = planning.get('danger_zone_m', 3.0)
        self.danger_pixels = int(math.ceil(danger_m * 100 / self.cm_per_pixel))

        self.distance_cost_weight = planning.get('distance_cost_weight', 0.3)

        self.inflated_grid = None
        self.distance_field = None

    def inflate_obstacles(self) -> List[List[bool]]:
        print(f"  膨胀半径: {self.inflation_pixels} 像素 ({self.inflation_pixels * self.cm_per_pixel / 100:.2f} m)")

        self.inflated_grid = [[True for _ in range(self.width)] for _ in range(self.height)]

        r = self.inflation_pixels
        inflate_offsets = []
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                if dv * dv + du * du <= r * r:
                    inflate_offsets.append((dv, du))

        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:
                    for dv, du in inflate_offsets:
                        nv, nu = v + dv, u + du
                        if 0 <= nv < self.height and 0 <= nu < self.width:
                            self.inflated_grid[nv][nu] = False

        original_obstacles = sum(1 for v in range(self.height)
                                  for u in range(self.width)
                                  if not self.original_grid[v][u])
        inflated_obstacles = sum(1 for v in range(self.height)
                                  for u in range(self.width)
                                  if not self.inflated_grid[v][u])

        print(f"  原始障碍格子: {original_obstacles}")
        print(f"  膨胀后障碍格子: {inflated_obstacles}")

        return self.inflated_grid

    def compute_distance_field(self) -> List[List[float]]:
        print(f"  危险区域半径: {self.danger_pixels} 像素 ({self.danger_pixels * self.cm_per_pixel / 100:.2f} m)")

        INF = float('inf')
        self.distance_field = [[INF for _ in range(self.width)] for _ in range(self.height)]

        queue = deque()
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:
                    self.distance_field[v][u] = 0
                    queue.append((v, u))

        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

        while queue:
            v, u = queue.popleft()
            current_dist = self.distance_field[v][u]
            if current_dist > self.danger_pixels:
                continue
            for dv, du, step_cost in directions:
                nv, nu = v + dv, u + du
                if 0 <= nv < self.height and 0 <= nu < self.width:
                    new_dist = current_dist + step_cost
                    if new_dist < self.distance_field[nv][nu]:
                        self.distance_field[nv][nu] = new_dist
                        queue.append((nv, nu))

        return self.distance_field

    def get_danger_penalty(self, v: int, u: int) -> float:
        if self.distance_field is None:
            return 0.0
        dist = self.distance_field[v][u]
        if dist >= self.danger_pixels:
            return 0.0
        return 1.0 - dist / self.danger_pixels

    def visualize_cspace(self, output_path: str):
        img = Image.new('RGB', (self.width, self.height))
        pixels = img.load()
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:
                    pixels[u, v] = (0, 0, 0)
                elif not self.inflated_grid[v][u]:
                    pixels[u, v] = (200, 50, 50)
                else:
                    dist = self.distance_field[v][u]
                    if dist < self.danger_pixels:
                        ratio = dist / self.danger_pixels
                        r = int(255 * (1 - ratio))
                        g = int(200 + 55 * ratio)
                        b = int(50 * ratio)
                        pixels[u, v] = (r, g, b)
                    else:
                        pixels[u, v] = (255, 255, 255)
        img.save(output_path)
        print(f"  配置空间可视化已保存至: {output_path}")


# ============================================================
# 坐标转换（与 v1 相同）
# ============================================================

class CoordinateTransformer:
    """坐标转换器：像素坐标 ↔ UE 世界坐标"""

    def __init__(self, config: dict):
        bounds = config['world_bounds']
        self.x_min = bounds['x_min']
        self.x_max = bounds['x_max']
        self.y_min = bounds['y_min']
        self.y_max = bounds['y_max']
        self.z_fixed = bounds['z_fixed']

        self.cm_per_pixel = config['resolution']['cm_per_pixel']

        img_size = config['image_size']
        self.width = img_size['width']
        self.height = img_size['height']

        coord_conv = config['coordinate_convention']
        self.flip_v = coord_conv.get('flip_v', False)

    def pixel_to_world(self, u: int, v: int) -> Tuple[float, float, float]:
        """像素坐标 → UE 世界坐标（取像素中心）"""
        if self.flip_v:
            v = self.height - 1 - v
        x = self.x_min + (u + 0.5) * self.cm_per_pixel
        y = self.y_min + (v + 0.5) * self.cm_per_pixel
        return (x, y, self.z_fixed)

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """UE 世界坐标 → 像素坐标"""
        u = int((x - self.x_min) / self.cm_per_pixel)
        v = int((y - self.y_min) / self.cm_per_pixel)
        if self.flip_v:
            v = self.height - 1 - v
        u = max(0, min(u, self.width - 1))
        v = max(0, min(v, self.height - 1))
        return (u, v)


# ============================================================
# 路径后处理工具（与 v1 相同）
# ============================================================

def smooth_path_rdp(path: List[Tuple[int, int]], epsilon: float = 1.0) -> List[Tuple[int, int]]:
    """Ramer-Douglas-Peucker 算法简化路径"""
    if len(path) < 3:
        return path

    def perpendicular_distance(point, line_start, line_end):
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)

    def rdp_recursive(points, epsilon):
        if len(points) < 3:
            return points
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                dmax = d
                index = i
        if dmax > epsilon:
            left = rdp_recursive(points[:index+1], epsilon)
            right = rdp_recursive(points[index:], epsilon)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]

    return rdp_recursive(path, epsilon)


def resample_path(world_path: List[Tuple[float, float, float]],
                  interval_cm: float = 200.0) -> List[Tuple[float, float, float]]:
    """按固定间距重采样路径"""
    if len(world_path) < 2:
        return world_path
    resampled = [world_path[0]]
    accumulated = 0.0
    for i in range(1, len(world_path)):
        dx = world_path[i][0] - world_path[i-1][0]
        dy = world_path[i][1] - world_path[i-1][1]
        dist = math.sqrt(dx**2 + dy**2)
        accumulated += dist
        if accumulated >= interval_cm:
            resampled.append(world_path[i])
            accumulated = 0.0
    if resampled[-1] != world_path[-1]:
        resampled.append(world_path[-1])
    return resampled


def compute_yaw(p1: Tuple[float, float, float],
                p2: Tuple[float, float, float]) -> float:
    """计算从 p1 指向 p2 的航向角（度）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def format_ue_commands(waypoints: List[dict]) -> List[str]:
    commands = []
    for wp in waypoints:
        cmd = f"SetPose:{wp['x']:.2f},{wp['y']:.2f},{wp['z']:.2f},{wp['pitch']:.2f},{wp['yaw']:.2f},{wp['roll']:.2f}"
        commands.append(cmd)
    return commands


def save_waypoints(waypoints: List[dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(waypoints, f, indent=2, ensure_ascii=False)
    print(f"Waypoints 已保存至: {output_path}")


def evaluate_path(path_pixels: List[Tuple[int, int]],
                  cspace: ConfigurationSpace,
                  transformer: CoordinateTransformer) -> dict:
    """评估路径质量"""
    metrics = {}

    world_path = [transformer.pixel_to_world(u, v) for u, v in path_pixels]
    total_length_cm = 0
    for i in range(1, len(world_path)):
        dx = world_path[i][0] - world_path[i-1][0]
        dy = world_path[i][1] - world_path[i-1][1]
        total_length_cm += math.sqrt(dx**2 + dy**2)
    metrics['path_length_m'] = total_length_cm / 100

    start_world = world_path[0]
    end_world = world_path[-1]
    direct_dist_cm = math.sqrt(
        (end_world[0] - start_world[0])**2 + (end_world[1] - start_world[1])**2
    )
    metrics['direct_distance_m'] = direct_dist_cm / 100
    metrics['path_efficiency'] = direct_dist_cm / total_length_cm if total_length_cm > 0 else 0

    min_clearance = float('inf')
    for u, v in path_pixels:
        dist = cspace.distance_field[v][u]
        if dist < min_clearance:
            min_clearance = dist
    metrics['min_clearance_m'] = min_clearance * cspace.cm_per_pixel / 100

    total_clearance = sum(cspace.distance_field[v][u] for u, v in path_pixels)
    avg_clearance = total_clearance / len(path_pixels) if path_pixels else 0
    metrics['avg_clearance_m'] = avg_clearance * cspace.cm_per_pixel / 100
    metrics['path_points'] = len(path_pixels)

    return metrics


# ============================================================
# SafeAStarPlanner（保留作为回退）
# ============================================================

class SafeAStarPlanner:
    """原始安全 A* 规划器，当 Hybrid A* 失败时使用"""

    NEIGHBORS_8 = [
        (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
        (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414),
    ]

    def __init__(self, cspace: ConfigurationSpace):
        self.cspace = cspace
        self.grid = cspace.inflated_grid
        self.height = cspace.height
        self.width = cspace.width
        self.lambda_weight = cspace.distance_cost_weight

    def is_valid(self, v: int, u: int) -> bool:
        if 0 <= v < self.height and 0 <= u < self.width:
            return self.grid[v][u]
        return False

    def heuristic(self, v1, u1, v2, u2) -> float:
        return math.sqrt((v2 - v1)**2 + (u2 - u1)**2)

    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_u, start_v = start
        goal_u, goal_v = goal

        if not self.is_valid(start_v, start_u):
            print(f"  错误：起点不可通行")
            return None
        if not self.is_valid(goal_v, goal_u):
            print(f"  错误：终点不可通行")
            return None

        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start_v, start_u))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {}
        g_score[(start_v, start_u)] = 0

        closed_set = set()
        iterations = 0
        max_iterations = self.width * self.height * 2

        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current_v, current_u = heapq.heappop(open_set)
            current = (current_v, current_u)

            if current_v == goal_v and current_u == goal_u:
                path = self._reconstruct_path(came_from, current)
                print(f"  A* 搜索完成：迭代 {iterations} 次，路径长度 {len(path)} 点")
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for dv, du, move_cost in self.NEIGHBORS_8:
                neighbor_v = current_v + dv
                neighbor_u = current_u + du
                neighbor = (neighbor_v, neighbor_u)

                if not self.is_valid(neighbor_v, neighbor_u):
                    continue
                if neighbor in closed_set:
                    continue
                if dv != 0 and du != 0:
                    if not self.is_valid(current_v + dv, current_u) or \
                       not self.is_valid(current_v, current_u + du):
                        continue

                danger_penalty = self.cspace.get_danger_penalty(neighbor_v, neighbor_u)
                step_cost = move_cost + self.lambda_weight * danger_penalty
                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor_v, neighbor_u, goal_v, goal_u)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor_v, neighbor_u))

        print(f"  A* 搜索失败：未找到路径，迭代 {iterations} 次")
        return None

    def _reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            v, u = current
            path.append((u, v))
            current = came_from[current]
        v, u = current
        path.append((u, v))
        path.reverse()
        return path


# ============================================================
# Hybrid A* 配置
# ============================================================

@dataclass
class HybridAStarConfig:
    """Hybrid A* 算法参数"""

    # 运动学参数
    r_min_m: float = 2.0          # 最小转弯半径（米）
    L_base_m: float = 1.0         # 基础步长（米）
    gamma_step: float = 0.5       # tanh 陡度参数
    d0_step_m: float = 5.0        # tanh 拐点距离（米）
    L_min_factor: float = 0.4     # 最小步长 = L_base * L_min_factor
    L_max_factor: float = 1.5     # 最大步长 = L_base * L_max_factor
    n_steerings: int = 5          # 转向原语数量
    allow_reverse: bool = False   # 是否允许倒退

    # 状态离散化
    n_theta_bins: int = 36        # 朝向离散化格子数（10°/格）

    # 代价权重
    lambda_danger: float = 0.3    # 危险惩罚权重
    w_heading: float = 0.5        # p₁：航向变化惩罚权重
    w_reversal: float = 2.0       # p₂：方向反转惩罚权重
    heading_penalty_threshold: float = math.pi / 4  # 开始惩罚的航向变化阈值

    # 目标容差
    goal_pos_tol_m: float = 1.0   # 位置容差（米）

    # 搜索限制
    max_iterations: int = 500_000

    # 梯度下降平滑参数
    w_smooth: float = 0.3         # 平滑代价权重
    w_obstacle: float = 0.5       # 障碍约束权重
    w_curvature: float = 0.2      # 曲率约束权重
    d_safe_m: float = 1.5         # 障碍安全距离目标（米）
    kappa_max: float = 0.5        # 最大曲率（= 1/r_min）
    smooth_iterations: int = 200  # 梯度下降迭代次数
    armijo_alpha: float = 0.3     # Armijo 充分下降常数
    armijo_beta: float = 0.5      # Armijo 步长缩减因子
    armijo_init_step: float = 0.1 # 初始步长

    @classmethod
    def from_config(cls, config: dict) -> 'HybridAStarConfig':
        ha = config.get('hybrid_astar', {})
        obj = cls()
        for k, v in ha.items():
            if hasattr(obj, k):
                setattr(obj, k, type(getattr(obj, k))(v))
        if 'kappa_max' not in ha:
            obj.kappa_max = 1.0 / obj.r_min_m
        return obj


# ============================================================
# Hybrid A* 节点
# ============================================================

class HybridAStarNode:
    """Hybrid A* 搜索节点，状态为 (x, y, θ)（米，弧度）"""

    __slots__ = ('x', 'y', 'theta', 'g', 'h', 'parent', 'is_forward')

    def __init__(self, x: float, y: float, theta: float,
                 g: float = 0.0, h: float = 0.0,
                 parent: Optional['HybridAStarNode'] = None,
                 is_forward: bool = True):
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

    def __lt__(self, other: 'HybridAStarNode') -> bool:
        return self.f < other.f


# ============================================================
# Dubins 曲线计算（公式 4.4）
# ============================================================

class DubinsPath:
    """
    计算两个位形之间的最短 Dubins 路径长度。
    6 种路径类型：LSL, RSR, LSR, RSL, RLR, LRL
    所有内部计算在归一化坐标（R=1）下进行，最后乘以 R。
    """

    @staticmethod
    def compute_min_length(x1: float, y1: float, t1: float,
                           x2: float, y2: float, t2: float,
                           R: float) -> float:
        """
        计算最短 Dubins 路径长度

        Args:
            x1, y1, t1: 起点位置（米）和朝向（弧度）
            x2, y2, t2: 终点位置和朝向
            R: 最小转弯半径（米）

        Returns:
            最短路径长度（米）
        """
        dx = x2 - x1
        dy = y2 - y1
        D = math.hypot(dx, dy)

        if D < 1e-9:
            d_theta = abs(_normalize_angle(t2 - t1))
            return R * min(d_theta, 2 * math.pi - d_theta)

        d = D / R
        angle = math.atan2(dy, dx)
        alpha = _mod2pi(t1 - angle)
        beta  = _mod2pi(t2 - angle)

        lengths = []
        for word in ('LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL'):
            L = DubinsPath._word_length(d, alpha, beta, word)
            if L is not None and L >= 0:
                lengths.append(L)

        if not lengths:
            return D  # 欧几里得回退（仍然可接受）

        return R * min(lengths)

    @staticmethod
    def _word_length(d: float, alpha: float, beta: float, word: str) -> Optional[float]:
        return {
            'LSL': DubinsPath._LSL,
            'RSR': DubinsPath._RSR,
            'LSR': DubinsPath._LSR,
            'RSL': DubinsPath._RSL,
            'RLR': DubinsPath._RLR,
            'LRL': DubinsPath._LRL,
        }[word](d, alpha, beta)

    @staticmethod
    def _LSL(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        p_sq = 2 + d*d - 2*math.cos(alpha - beta) + 2*d*(sa - sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(cb - ca, d + sa - sb)
        t = _mod2pi(-alpha + tmp)
        q = _mod2pi(beta - tmp)
        return t + p + q

    @staticmethod
    def _RSR(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        p_sq = 2 + d*d - 2*math.cos(alpha - beta) + 2*d*(sb - sa)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca - cb, d - sa + sb)
        t = _mod2pi(alpha - tmp)
        q = _mod2pi(-beta + tmp)
        return t + p + q

    @staticmethod
    def _LSR(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        p_sq = -2 + d*d + 2*math.cos(alpha - beta) + 2*d*(sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        t = _mod2pi(-alpha + tmp)
        q = _mod2pi(-beta + tmp)
        return t + p + q

    @staticmethod
    def _RSL(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        p_sq = -2 + d*d + 2*math.cos(alpha - beta) - 2*d*(sa + sb)
        if p_sq < 0:
            return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        t = _mod2pi(alpha - tmp)
        q = _mod2pi(beta - tmp)
        return t + p + q

    @staticmethod
    def _RLR(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        tmp = (6.0 - d*d + 2*math.cos(alpha - beta) + 2*d*(sa - sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2*math.pi - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
        q = _mod2pi(alpha - beta - t + _mod2pi(p))
        return t + p + q

    @staticmethod
    def _LRL(d, alpha, beta):
        sa, ca = math.sin(alpha), math.cos(alpha)
        sb, cb = math.sin(beta),  math.cos(beta)
        tmp = (6.0 - d*d + 2*math.cos(alpha - beta) + 2*d*(-sa + sb)) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2*math.pi - math.acos(tmp))
        t = _mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
        return t + p + q


# ============================================================
# Hybrid A* 规划器
# ============================================================

class HybridAStarPlanner:
    """
    基于运动学约束的 Hybrid A* 路径规划器

    状态空间：(x, y, θ) 连续位置和朝向
    搜索空间：离散化的 (u, v, θ_bin) 用于 closed set
    代价函数：f(n) = g(n) + h_dubins(n) + p₁(航向变化) + p₂(方向反转)
    """

    def __init__(self, cspace: ConfigurationSpace,
                 cfg: HybridAStarConfig,
                 transformer: CoordinateTransformer):
        self.cspace = cspace
        self.cfg = cfg
        self.transformer = transformer
        self.pixel_size_m = cspace.cm_per_pixel / 100.0

    def _meters_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return self.transformer.world_to_pixel(x_m * 100.0, y_m * 100.0)

    def _is_valid_pixel(self, u: int, v: int) -> bool:
        if 0 <= v < self.cspace.height and 0 <= u < self.cspace.width:
            return self.cspace.inflated_grid[v][u]
        return False

    def _dynamic_step_length(self, dist_to_goal_m: float) -> float:
        """自适应步长（公式 4.3）"""
        cfg = self.cfg
        k = 0.5 * (1.0 + math.tanh(cfg.gamma_step * (dist_to_goal_m - cfg.d0_step_m)))
        scale = cfg.L_min_factor + (cfg.L_max_factor - cfg.L_min_factor) * k
        return cfg.L_base_m * scale

    def _get_curvatures(self) -> List[float]:
        """生成转向曲率原语列表"""
        kmax = 1.0 / self.cfg.r_min_m
        n = self.cfg.n_steerings
        if n == 1:
            return [0.0]
        return [-kmax + (2 * kmax * i) / (n - 1) for i in range(n)]

    def _apply_motion(self, x: float, y: float, theta: float,
                      L: float, k: float) -> Tuple[float, float, float]:
        """
        应用运动学模型（公式 4.2）
        L: 带符号的弧长（正=前进，负=后退）
        k: 曲率（正=左转，负=右转）
        """
        EPSILON = 1e-6
        delta_theta = k * L

        if abs(delta_theta) < EPSILON:
            return (x + L * math.cos(theta),
                    y + L * math.sin(theta),
                    theta)

        r = L / delta_theta
        x_new = x + r * (math.sin(theta + delta_theta) - math.sin(theta))
        y_new = y - r * (math.cos(theta + delta_theta) - math.cos(theta))
        theta_new = _normalize_angle(theta + delta_theta)
        return x_new, y_new, theta_new

    def _arc_collision_free(self, x0: float, y0: float, theta0: float,
                             L: float, k: float) -> bool:
        """沿弧段采样检查碰撞"""
        L_abs = abs(L)
        n_check = max(3, int(L_abs / (0.5 * self.pixel_size_m)))
        for i in range(1, n_check + 1):
            frac = i / n_check
            xi, yi, _ = self._apply_motion(x0, y0, theta0, L * frac, k)
            u, v = self._meters_to_pixel(xi, yi)
            if not self._is_valid_pixel(u, v):
                return False
        return True

    def _discretize_state(self, x_m: float, y_m: float,
                          theta: float) -> Tuple[int, int, int]:
        """离散化连续状态用于 closed set"""
        u, v = self._meters_to_pixel(x_m, y_m)
        theta_norm = (_normalize_angle(theta) + math.pi) / (2 * math.pi)
        theta_bin = int(theta_norm * self.cfg.n_theta_bins) % self.cfg.n_theta_bins
        return (u, v, theta_bin)

    def _heuristic_dubins(self, x: float, y: float, theta: float,
                          gx: float, gy: float, g_theta: float) -> float:
        """Dubins 曲线启发式（公式 4.4），可接受且比欧几里得更紧"""
        return DubinsPath.compute_min_length(x, y, theta, gx, gy, g_theta, self.cfg.r_min_m)

    def search(self, start_pixel: Tuple[int, int],
               goal_pixel: Tuple[int, int],
               start_theta: Optional[float] = None) -> Optional[List[HybridAStarNode]]:
        """
        执行 Hybrid A* 搜索

        Args:
            start_pixel: (u, v) 起点像素坐标
            goal_pixel:  (u, v) 终点像素坐标
            start_theta: 起始朝向（弧度），None 表示自动指向终点

        Returns:
            HybridAStarNode 列表（路径），或 None（无解）
        """
        su, sv = start_pixel
        gu, gv = goal_pixel

        # 转换为米
        sx_cm, sy_cm, _ = self.transformer.pixel_to_world(su, sv)
        gx_cm, gy_cm, _ = self.transformer.pixel_to_world(gu, gv)
        sx, sy = sx_cm / 100.0, sy_cm / 100.0
        gx, gy = gx_cm / 100.0, gy_cm / 100.0

        # 起始朝向
        if start_theta is None:
            start_theta = math.atan2(gy - sy, gx - sx)

        # 起终点有效性检查
        if not self._is_valid_pixel(su, sv):
            if not (0 <= sv < self.cspace.height and 0 <= su < self.cspace.width
                    and self.cspace.original_grid[sv][su]):
                print(f"  错误：起点 ({su}, {sv}) 不可通行")
                return None
            print(f"  警告：起点在膨胀区域内，继续搜索...")

        if not self._is_valid_pixel(gu, gv):
            if not (0 <= gv < self.cspace.height and 0 <= gu < self.cspace.width
                    and self.cspace.original_grid[gv][gu]):
                print(f"  错误：终点 ({gu}, {gv}) 不可通行")
                return None
            print(f"  警告：终点在膨胀区域内，继续搜索...")

        # 初始节点
        goal_theta = math.atan2(gy - sy, gx - sx)
        h0 = self._heuristic_dubins(sx, sy, start_theta, gx, gy, goal_theta)
        start_node = HybridAStarNode(x=sx, y=sy, theta=start_theta, g=0.0, h=h0)

        # 优先队列：(f, counter, node)
        open_heap: List = []
        counter = 0
        heapq.heappush(open_heap, (start_node.f, counter, start_node))

        # 最优 g 值字典
        g_best: Dict[Tuple[int, int, int], float] = {}
        g_best[self._discretize_state(sx, sy, start_theta)] = 0.0

        closed_set: set = set()
        curvatures = self._get_curvatures()
        iterations = 0

        while open_heap and iterations < self.cfg.max_iterations:
            iterations += 1
            _, _, node = heapq.heappop(open_heap)

            # 目标检测
            dist = math.hypot(node.x - gx, node.y - gy)
            if dist <= self.cfg.goal_pos_tol_m:
                path = self._reconstruct_path(node)
                print(f"  Hybrid A* 完成：迭代 {iterations} 次，路径 {len(path)} 节点")
                return path

            disc_key = self._discretize_state(node.x, node.y, node.theta)
            if disc_key in closed_set:
                continue
            closed_set.add(disc_key)

            # 动态步长
            dist_to_goal = math.hypot(gx - node.x, gy - node.y)
            L = self._dynamic_step_length(dist_to_goal)

            # 展开邻居
            directions = [True]
            if self.cfg.allow_reverse:
                directions.append(False)

            for is_forward in directions:
                L_signed = L if is_forward else -L

                for k in curvatures:
                    xn, yn, tn = self._apply_motion(
                        node.x, node.y, node.theta, L_signed, k)

                    un, vn = self._meters_to_pixel(xn, yn)
                    if not self._is_valid_pixel(un, vn):
                        continue
                    if not self._arc_collision_free(
                            node.x, node.y, node.theta, L_signed, k):
                        continue

                    # 弧长代价 + 安全惩罚
                    arc_cost = abs(L_signed)
                    danger_p = (self.cspace.get_danger_penalty(vn, un)
                                * self.cfg.lambda_danger)

                    # p₁：航向变化惩罚
                    delta_heading = abs(_normalize_angle(tn - node.theta))
                    p1 = 0.0
                    if delta_heading > self.cfg.heading_penalty_threshold:
                        p1 = self.cfg.w_heading * (
                            delta_heading - self.cfg.heading_penalty_threshold)

                    # p₂：方向反转惩罚
                    p2 = 0.0
                    if node.parent is not None and is_forward != node.is_forward:
                        p2 = self.cfg.w_reversal

                    g_new = node.g + arc_cost + danger_p + p1 + p2

                    disc_n = self._discretize_state(xn, yn, tn)
                    if disc_n in closed_set:
                        continue
                    if disc_n in g_best and g_best[disc_n] <= g_new:
                        continue

                    g_best[disc_n] = g_new

                    # 目标朝向 = 从新节点指向目标
                    g_theta = math.atan2(gy - yn, gx - xn)
                    h_new = self._heuristic_dubins(xn, yn, tn, gx, gy, g_theta)

                    child = HybridAStarNode(
                        x=xn, y=yn, theta=tn,
                        g=g_new, h=h_new,
                        parent=node,
                        is_forward=is_forward
                    )
                    counter += 1
                    heapq.heappush(open_heap, (child.f, counter, child))

        print(f"  Hybrid A* 搜索失败：达到最大迭代 {iterations} 次")
        return None

    def _reconstruct_path(self, node: HybridAStarNode) -> List[HybridAStarNode]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path


# ============================================================
# 梯度下降路径平滑器（公式 4.24）
# ============================================================

class GradientDescentSmoother:
    """
    基于梯度下降的路径平滑优化器

    代价函数：F(X) = F_s（平滑度）+ F_o（障碍约束）+ F_r（曲率约束）
    使用 Armijo 回溯线搜索调整步长（公式 4.34）
    """

    def __init__(self, path_meters: List[Tuple[float, float]],
                 cspace: ConfigurationSpace,
                 transformer: CoordinateTransformer,
                 cfg: HybridAStarConfig):
        self.path_meters = path_meters
        self.cspace = cspace
        self.transformer = transformer
        self.cfg = cfg
        self.pixel_size_m = cspace.cm_per_pixel / 100.0

    def _meters_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return self.transformer.world_to_pixel(x_m * 100.0, y_m * 100.0)

    def _dist_at_m(self, x_m: float, y_m: float) -> float:
        """获取某点到最近障碍的距离（米）"""
        u, v = self._meters_to_pixel(x_m, y_m)
        u = max(0, min(u, self.cspace.width - 1))
        v = max(0, min(v, self.cspace.height - 1))
        d_pix = self.cspace.distance_field[v][u]
        if d_pix == float('inf'):
            return self.cfg.d_safe_m * 10  # 远离障碍
        return d_pix * self.pixel_size_m

    def _is_valid_m(self, x_m: float, y_m: float) -> bool:
        u, v = self._meters_to_pixel(x_m, y_m)
        if 0 <= v < self.cspace.height and 0 <= u < self.cspace.width:
            return self.cspace.inflated_grid[v][u]
        return False

    def _kappa(self, X: np.ndarray, i: int) -> float:
        """Menger 离散曲率"""
        A = X[i] - X[i-1]
        B = X[i+1] - X[i]
        la = float(np.linalg.norm(A))
        lb = float(np.linalg.norm(B))
        if la < 1e-10 or lb < 1e-10:
            return 0.0
        cross = float(A[0]*B[1] - A[1]*B[0])
        chord = X[i+1] - X[i-1]
        lc = float(np.linalg.norm(chord))
        if lc < 1e-10:
            return 0.0
        return 2.0 * abs(cross) / (la * lb * lc)

    def _F_s(self, X: np.ndarray) -> float:
        """平滑代价：惩罚二阶差分（曲率）"""
        cost = 0.0
        for i in range(1, len(X) - 1):
            d = X[i+1] - 2*X[i] + X[i-1]
            cost += float(np.dot(d, d))
        return self.cfg.w_smooth * cost

    def _F_o(self, X: np.ndarray) -> float:
        """障碍安全代价：保持安全距离"""
        cost = 0.0
        for i in range(1, len(X) - 1):
            dist = self._dist_at_m(X[i, 0], X[i, 1])
            pen = max(0.0, self.cfg.d_safe_m - dist)
            cost += pen * pen
        return self.cfg.w_obstacle * cost

    def _F_r(self, X: np.ndarray) -> float:
        """曲率约束代价：惩罚超过最大曲率的弯曲"""
        cost = 0.0
        for i in range(1, len(X) - 1):
            kappa = self._kappa(X, i)
            excess = max(0.0, kappa - self.cfg.kappa_max)
            cost += excess * excess
        return self.cfg.w_curvature * cost

    def _total_cost(self, X: np.ndarray) -> float:
        return self._F_s(X) + self._F_o(X) + self._F_r(X)

    def _grad_smoothness(self, X: np.ndarray) -> np.ndarray:
        """平滑项的解析梯度"""
        N = len(X)
        grad = np.zeros_like(X)
        for j in range(1, N - 1):
            d_jm1 = (X[j] - 2*X[j-1] + X[j-2]) if j >= 2 else np.zeros(2)
            d_j   = X[j+1] - 2*X[j] + X[j-1]
            d_jp1 = (X[j+2] - 2*X[j+1] + X[j]) if j <= N-3 else np.zeros(2)
            grad[j] = 2.0 * self.cfg.w_smooth * (d_jm1 - 2*d_j + d_jp1)
        return grad

    def _grad_obstacle_numerical(self, X: np.ndarray) -> np.ndarray:
        """障碍项的数值梯度（有限差分）"""
        N = len(X)
        grad = np.zeros_like(X)
        eps = self.pixel_size_m
        d_safe = self.cfg.d_safe_m

        for i in range(1, N - 1):
            d = self._dist_at_m(X[i, 0], X[i, 1])
            pen = max(0.0, d_safe - d)
            if pen > 0:
                dx = (self._dist_at_m(X[i,0] + eps, X[i,1]) -
                      self._dist_at_m(X[i,0] - eps, X[i,1])) / (2*eps)
                dy = (self._dist_at_m(X[i,0], X[i,1] + eps) -
                      self._dist_at_m(X[i,0], X[i,1] - eps)) / (2*eps)
                grad[i] = -2.0 * self.cfg.w_obstacle * pen * np.array([dx, dy])
        return grad

    def _grad_curvature_numerical(self, X: np.ndarray) -> np.ndarray:
        """曲率约束的数值梯度（有限差分）"""
        N = len(X)
        grad = np.zeros_like(X)
        eps = 1e-4

        for i in range(1, N - 1):
            if self._kappa(X, i) <= self.cfg.kappa_max:
                continue
            for j_dim in range(2):
                Xp = X.copy(); Xp[i, j_dim] += eps
                Xm = X.copy(); Xm[i, j_dim] -= eps
                grad[i, j_dim] = (self._F_r(Xp) - self._F_r(Xm)) / (2 * eps)
        return grad

    def _armijo_step(self, X: np.ndarray, grad: np.ndarray,
                     current_cost: float) -> float:
        """Armijo 回溯线搜索（公式 4.34）"""
        c = self.cfg.armijo_alpha
        beta = self.cfg.armijo_beta
        alpha = self.cfg.armijo_init_step
        grad_norm_sq = float(np.sum(grad[1:-1] ** 2))

        if grad_norm_sq < 1e-15:
            return alpha

        for _ in range(30):
            X_new = X.copy()
            X_new[1:-1] -= alpha * grad[1:-1]
            new_cost = self._total_cost(X_new)
            if new_cost <= current_cost - c * alpha * grad_norm_sq:
                return alpha
            alpha *= beta

        return alpha

    def smooth(self) -> List[Tuple[float, float]]:
        """执行梯度下降路径平滑，返回平滑后的 (x_m, y_m) 列表"""
        if len(self.path_meters) < 3:
            return list(self.path_meters)

        X = np.array([[p[0], p[1]] for p in self.path_meters], dtype=float)
        X_orig = X.copy()

        for iteration in range(self.cfg.smooth_iterations):
            cost = self._total_cost(X)

            grad = (self._grad_smoothness(X) +
                    self._grad_obstacle_numerical(X) +
                    self._grad_curvature_numerical(X))

            grad_max = float(np.max(np.abs(grad[1:-1]))) if len(X) > 2 else 0.0
            if grad_max < 1e-7:
                print(f"    梯度下降收敛于第 {iteration} 次迭代")
                break

            alpha = self._armijo_step(X, grad, cost)

            X_new = X.copy()
            X_new[1:-1] -= alpha * grad[1:-1]

            # 碰撞验证
            valid = all(self._is_valid_m(X_new[i, 0], X_new[i, 1])
                        for i in range(1, len(X_new) - 1))
            if valid:
                X = X_new
            else:
                # 尝试更保守的步长
                X_new2 = X.copy()
                X_new2[1:-1] -= (alpha * 0.1) * grad[1:-1]
                if all(self._is_valid_m(X_new2[i, 0], X_new2[i, 1])
                       for i in range(1, len(X_new2) - 1)):
                    X = X_new2

        # 最终碰撞验证
        for i in range(1, len(X) - 1):
            if not self._is_valid_m(X[i, 0], X[i, 1]):
                print("    警告：平滑路径有碰撞，回退至平滑前")
                return [(float(X_orig[i, 0]), float(X_orig[i, 1]))
                        for i in range(len(X_orig))]

        return [(float(X[i, 0]), float(X[i, 1])) for i in range(len(X))]


# ============================================================
# 可视化
# ============================================================

def visualize_path_on_cspace(cspace: ConfigurationSpace,
                              path: List[Tuple[int, int]],
                              output_path: str,
                              start: Tuple[int, int],
                              goal: Tuple[int, int]):
    """在配置空间上绘制路径"""
    width, height = cspace.width, cspace.height
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    for v in range(height):
        for u in range(width):
            if not cspace.original_grid[v][u]:
                pixels[u, v] = (30, 30, 30)
            elif not cspace.inflated_grid[v][u]:
                pixels[u, v] = (100, 40, 40)
            else:
                dist = cspace.distance_field[v][u]
                if dist < cspace.danger_pixels:
                    ratio = dist / cspace.danger_pixels
                    r = int(180 * (1 - ratio) + 50)
                    g = int(150 * ratio + 50)
                    b = 50
                    pixels[u, v] = (r, g, b)
                else:
                    pixels[u, v] = (220, 220, 220)

    draw = ImageDraw.Draw(img)

    if len(path) > 1:
        for i in range(len(path) - 1):
            draw.line([path[i], path[i+1]], fill=(0, 150, 255), width=3)
    for p in path:
        r = 2
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=(0, 200, 255))

    r = 6
    draw.ellipse([start[0]-r, start[1]-r, start[0]+r, start[1]+r], fill=(0, 255, 0))
    draw.ellipse([goal[0]-r, goal[1]-r, goal[0]+r, goal[1]+r], fill=(255, 0, 0))

    img.save(output_path)
    print(f"路径可视化已保存至: {output_path}")


# ============================================================
# Hybrid A* Waypoint 生成
# ============================================================

def ha_path_to_pixels(ha_path: List[HybridAStarNode],
                      transformer: CoordinateTransformer) -> List[Tuple[int, int]]:
    """将 Hybrid A* 节点路径转换为像素坐标"""
    result = []
    for node in ha_path:
        u, v = transformer.world_to_pixel(node.x * 100.0, node.y * 100.0)
        result.append((u, v))
    return result


def generate_waypoints_hybrid(ha_path: List[HybridAStarNode],
                               cspace: ConfigurationSpace,
                               transformer: CoordinateTransformer,
                               cfg: HybridAStarConfig,
                               resample_interval_cm: float = 200.0) -> List[dict]:
    """
    从 Hybrid A* 路径生成 UE waypoints

    流程：节点路径 → RDP 简化 → 梯度下降平滑 → 重采样 → 航向计算
    输出格式与 v1 完全兼容：{x, y, z, pitch, yaw, roll}（厘米）
    """
    # 1. 转为像素坐标，RDP 简化
    path_pixels = ha_path_to_pixels(ha_path, transformer)
    path_simplified_pix = smooth_path_rdp(path_pixels, epsilon=2.0)
    print(f"    原始节点: {len(path_pixels)}，RDP 后: {len(path_simplified_pix)} 点")

    # 2. 转回米坐标
    path_m: List[Tuple[float, float]] = []
    for u, v in path_simplified_pix:
        x_cm, y_cm, _ = transformer.pixel_to_world(u, v)
        path_m.append((x_cm / 100.0, y_cm / 100.0))

    # 3. 梯度下降平滑
    smoother = GradientDescentSmoother(path_m, cspace, transformer, cfg)
    smoothed_m = smoother.smooth()

    # 4. 转为 UE 世界坐标（厘米）
    z_fixed = transformer.z_fixed
    world_path = [(x * 100.0, y * 100.0, z_fixed) for x, y in smoothed_m]

    # 5. 重采样
    world_path = resample_path(world_path, resample_interval_cm)

    # 6. 计算航向，构建 waypoints
    waypoints = []
    for i, (x, y, z) in enumerate(world_path):
        if i < len(world_path) - 1:
            yaw = compute_yaw(world_path[i], world_path[i+1])
        else:
            yaw = waypoints[-1]['yaw'] if waypoints else 0.0
        waypoints.append({
            'x': x, 'y': y, 'z': z,
            'pitch': 0.0, 'yaw': yaw, 'roll': 0.0
        })

    return waypoints


# ============================================================
# 主函数
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(project_dir, 'config')
    result_dir = os.path.join(project_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    config_path = os.path.join(config_dir, 'grid_config.json')
    image_path  = os.path.join(result_dir, 'occupancy_grid.png')

    print("=" * 60)
    print("Hybrid A* 路径规划 v2（运动学约束 + 梯度下降平滑）")
    print("=" * 60)

    # ========== 1. 加载配置 ==========
    config = load_config(config_path)
    print(f"\n[1] 加载配置: {config_path}")

    threshold = config['value_convention']['threshold']
    grid = load_occupancy_grid(image_path, threshold)
    print(f"    栅格尺寸: {len(grid[0])} x {len(grid)}")

    # ========== 2. 构建配置空间 ==========
    print(f"\n[2] 构建配置空间...")
    cspace = ConfigurationSpace(grid, config)
    cspace.inflate_obstacles()
    cspace.compute_distance_field()

    cspace_vis_path = os.path.join(result_dir, 'cspace_visualization.png')
    cspace.visualize_cspace(cspace_vis_path)

    # ========== 3. 坐标转换器 ==========
    transformer = CoordinateTransformer(config)

    # ========== 4. 起终点 ==========
    test_points = config.get('test_points', {})
    start_pixel = tuple(test_points.get('start_pixel', [200, 10]))
    goal_pixel  = tuple(test_points.get('end_pixel',   [190, 220]))

    print(f"\n[3] 起终点设置:")
    print(f"    起点像素: {start_pixel}")
    print(f"    终点像素: {goal_pixel}")

    # ========== 5. Hybrid A* 参数 ==========
    ha_cfg = HybridAStarConfig.from_config(config)
    print(f"\n[4] Hybrid A* 参数:")
    print(f"    最小转弯半径: {ha_cfg.r_min_m} m")
    print(f"    基础步长:     {ha_cfg.L_base_m} m")
    print(f"    转向原语数:   {ha_cfg.n_steerings}")
    print(f"    朝向分格数:   {ha_cfg.n_theta_bins}")
    print(f"    平滑迭代数:   {ha_cfg.smooth_iterations}")

    # ========== 6. Hybrid A* 搜索 ==========
    print(f"\n[5] 执行 Hybrid A* 搜索...")
    ha_planner = HybridAStarPlanner(cspace, ha_cfg, transformer)
    ha_path = ha_planner.search(start_pixel, goal_pixel)

    # 若失败，回退到 v1 SafeAStarPlanner
    use_fallback = ha_path is None
    if use_fallback:
        print("\n  Hybrid A* 失败，回退至 SafeAStarPlanner...")

        # 尝试缩小转弯半径再搜索一次
        ha_cfg_retry = HybridAStarConfig.from_config(config)
        ha_cfg_retry.r_min_m = ha_cfg.r_min_m / 2.0
        ha_cfg_retry.kappa_max = 1.0 / ha_cfg_retry.r_min_m
        print(f"  重试：r_min = {ha_cfg_retry.r_min_m} m...")
        ha_planner2 = HybridAStarPlanner(cspace, ha_cfg_retry, transformer)
        ha_path = ha_planner2.search(start_pixel, goal_pixel)
        if ha_path is not None:
            ha_cfg = ha_cfg_retry
            use_fallback = False
            print("  重试成功！")

    if use_fallback:
        print("\n[5b] 执行 SafeAStarPlanner (v1 回退)...")
        v1_planner = SafeAStarPlanner(cspace)
        path_pixels_v1 = v1_planner.search(start_pixel, goal_pixel)
        if path_pixels_v1 is None:
            print("\n未找到有效路径！")
            print("建议：减小 inflation_radius_m 或 safety_margin_m")
            return

        path_simplified = smooth_path_rdp(path_pixels_v1, epsilon=2.0)
        waypoints = []
        world_path = [transformer.pixel_to_world(u, v) for u, v in path_simplified]
        world_path = resample_path(world_path, 200.0)
        for i, (x, y, z) in enumerate(world_path):
            yaw = compute_yaw(world_path[i], world_path[i+1]) if i < len(world_path)-1 \
                  else (waypoints[-1]['yaw'] if waypoints else 0.0)
            waypoints.append({'x': x, 'y': y, 'z': z,
                               'pitch': 0.0, 'yaw': yaw, 'roll': 0.0})
        eval_pixels = path_pixels_v1
    else:
        # ========== 7. 路径平滑与 Waypoint 生成 ==========
        print(f"\n[6] 梯度下降路径平滑与 Waypoint 生成...")
        waypoints = generate_waypoints_hybrid(
            ha_path, cspace, transformer, ha_cfg, resample_interval_cm=200.0)
        eval_pixels = ha_path_to_pixels(ha_path, transformer)

    print(f"    最终 waypoints: {len(waypoints)} 个")

    # ========== 8. 保存结果 ==========
    print(f"\n[7] 保存结果...")

    vis_path = os.path.join(result_dir, 'path_result_hybrid.png')
    visualize_path_on_cspace(cspace, eval_pixels, vis_path, start_pixel, goal_pixel)

    waypoints_path = os.path.join(result_dir, 'waypoints.json')
    save_waypoints(waypoints, waypoints_path)

    commands = format_ue_commands(waypoints)
    commands_path = os.path.join(result_dir, 'ue_commands.txt')
    with open(commands_path, 'w') as f:
        f.write('\n'.join(commands))
    print(f"UE 指令已保存至: {commands_path}")

    # ========== 9. 路径评估 ==========
    print(f"\n[8] 路径评估指标:")
    print("-" * 40)
    metrics = evaluate_path(eval_pixels, cspace, transformer)
    print(f"    路径总长度:     {metrics['path_length_m']:.2f} m")
    print(f"    直线距离:       {metrics['direct_distance_m']:.2f} m")
    print(f"    路径效率:       {metrics['path_efficiency']*100:.1f}%")
    print(f"    最小离障距离:   {metrics['min_clearance_m']:.2f} m")
    print(f"    平均离障距离:   {metrics['avg_clearance_m']:.2f} m")
    print(f"    路径节点数:     {metrics['path_points']}")

    # ========== 10. 指令预览 ==========
    print(f"\n[9] UE 指令预览（前5条）:")
    for cmd in commands[:5]:
        print(f"    {cmd}")
    if len(commands) > 5:
        print(f"    ... (共 {len(commands)} 条)")

    print("\n" + "=" * 60)
    print("规划完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
