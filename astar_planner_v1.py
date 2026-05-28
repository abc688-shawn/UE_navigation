"""
A* 路径规划算法 - 增强版

功能：
1. 配置空间膨胀（C-space inflation）：基于航行器尺寸膨胀障碍物
2. 距离场计算：计算每个格子到最近障碍的距离
3. 安全代价函数：A* 代价 = 移动距离 + λ * 危险惩罚
4. 路径后处理（平滑、重采样）
5. 完整的可视化输出

代价函数设计：
  g(n) = 累计移动距离 + λ * Σ danger_penalty(cell)
  
  danger_penalty(cell) = max(0, 1 - dist_to_obstacle / danger_zone)
  
  这样：
  - 在障碍物边缘（dist=0）惩罚最大=1
  - 在 danger_zone 边缘惩罚=0
  - 超出 danger_zone 无惩罚
"""

import json
import heapq
import math
from collections import deque
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Dict


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
        grid: 2D list, grid[v][u]
              True = 可通行 (free)
              False = 障碍 (occupied)
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
# 配置空间处理（步骤二）
# ============================================================

class ConfigurationSpace:
    """
    配置空间处理器
    
    功能：
    1. 障碍物膨胀（基于航行器尺寸）
    2. 距离场计算（用于安全代价）
    """
    
    def __init__(self, grid: List[List[bool]], config: dict):
        """
        Args:
            grid: 原始占据栅格, grid[v][u], True=free, False=occupied
            config: 配置字典
        """
        self.original_grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        
        # 分辨率
        self.cm_per_pixel = config['resolution']['cm_per_pixel']
        
        # 规划参数
        planning = config.get('planning', {})
        
        # 膨胀半径（米 -> 像素）
        inflation_m = planning.get('inflation_radius_m', 1.0)
        safety_m = planning.get('safety_margin_m', 0.5)
        total_inflation_m = inflation_m + safety_m
        self.inflation_pixels = int(math.ceil(total_inflation_m * 100 / self.cm_per_pixel))
        
        # 危险区域（米 -> 像素）
        danger_m = planning.get('danger_zone_m', 3.0)
        self.danger_pixels = int(math.ceil(danger_m * 100 / self.cm_per_pixel))
        
        # 安全代价权重
        self.distance_cost_weight = planning.get('distance_cost_weight', 0.3)
        
        # 初始化
        self.inflated_grid = None
        self.distance_field = None
        
    def inflate_obstacles(self) -> List[List[bool]]:
        """
        对障碍物进行膨胀，生成配置空间
        
        使用形态学膨胀：以每个障碍格子为中心，将半径内的格子都标记为障碍
        
        Returns:
            inflated_grid: 膨胀后的栅格
        """
        print(f"  膨胀半径: {self.inflation_pixels} 像素 ({self.inflation_pixels * self.cm_per_pixel / 100:.2f} m)")
        
        # 创建膨胀后的栅格（初始化为全部可通行）
        self.inflated_grid = [[True for _ in range(self.width)] for _ in range(self.height)]
        
        # 预计算圆形膨胀模板
        r = self.inflation_pixels
        inflate_offsets = []
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                if dv * dv + du * du <= r * r:
                    inflate_offsets.append((dv, du))
        
        # 对每个障碍格子进行膨胀
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:  # 原始障碍
                    for dv, du in inflate_offsets:
                        nv, nu = v + dv, u + du
                        if 0 <= nv < self.height and 0 <= nu < self.width:
                            self.inflated_grid[nv][nu] = False
        
        # 统计
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
        """
        计算距离场：每个格子到最近障碍（膨胀前原始障碍）的距离
        
        使用 BFS 实现，得到准确的欧几里得距离近似
        
        Returns:
            distance_field: 2D list, distance_field[v][u] = 到最近障碍的像素距离
        """
        print(f"  危险区域半径: {self.danger_pixels} 像素 ({self.danger_pixels * self.cm_per_pixel / 100:.2f} m)")
        
        # 初始化距离场（无穷大）
        INF = float('inf')
        self.distance_field = [[INF for _ in range(self.width)] for _ in range(self.height)]
        
        # BFS 队列：从所有原始障碍格子开始
        queue = deque()
        
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:  # 原始障碍
                    self.distance_field[v][u] = 0
                    queue.append((v, u))
        
        # 8邻接方向
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]
        
        # BFS 扩展
        while queue:
            v, u = queue.popleft()
            current_dist = self.distance_field[v][u]
            
            # 只计算到 danger_zone 范围
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
        """
        获取某个格子的危险惩罚值
        
        penalty = max(0, 1 - dist / danger_zone)
        
        Returns:
            0.0 ~ 1.0 的惩罚值，越接近障碍值越大
        """
        if self.distance_field is None:
            return 0.0
        
        dist = self.distance_field[v][u]
        if dist >= self.danger_pixels:
            return 0.0
        
        # 线性衰减
        return 1.0 - dist / self.danger_pixels
    
    def visualize_cspace(self, output_path: str):
        """可视化配置空间和距离场"""
        # 创建 RGB 图像
        img = Image.new('RGB', (self.width, self.height))
        pixels = img.load()
        
        for v in range(self.height):
            for u in range(self.width):
                if not self.original_grid[v][u]:
                    # 原始障碍：黑色
                    pixels[u, v] = (0, 0, 0)
                elif not self.inflated_grid[v][u]:
                    # 膨胀区域：红色
                    pixels[u, v] = (200, 50, 50)
                else:
                    # 可通行区域：根据距离场着色
                    dist = self.distance_field[v][u]
                    if dist < self.danger_pixels:
                        # 危险区域：黄色渐变到绿色
                        ratio = dist / self.danger_pixels
                        r = int(255 * (1 - ratio))
                        g = int(200 + 55 * ratio)
                        b = int(50 * ratio)
                        pixels[u, v] = (r, g, b)
                    else:
                        # 安全区域：白色
                        pixels[u, v] = (255, 255, 255)
        
        img.save(output_path)
        print(f"  配置空间可视化已保存至: {output_path}")


# ============================================================
# 坐标转换
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
        z = self.z_fixed
        
        return (x, y, z)
    
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
# A* 算法核心（步骤三）
# ============================================================

class SafeAStarPlanner:
    """
    安全 A* 路径规划器
    
    代价函数: g(n) = 移动距离 + λ * Σ danger_penalty
    启发式: h(n) = 欧几里得距离（admissible）
    """
    
    # 8邻接：(dv, du, cost)
    NEIGHBORS_8 = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, 1.414),
        (-1,  1, 1.414),
        ( 1, -1, 1.414),
        ( 1,  1, 1.414),
    ]
    
    def __init__(self, cspace: ConfigurationSpace):
        """
        Args:
            cspace: 配置空间对象
        """
        self.cspace = cspace
        self.grid = cspace.inflated_grid  # 使用膨胀后的栅格
        self.height = cspace.height
        self.width = cspace.width
        self.lambda_weight = cspace.distance_cost_weight
    
    def is_valid(self, v: int, u: int) -> bool:
        """检查坐标是否有效且可通行"""
        if 0 <= v < self.height and 0 <= u < self.width:
            return self.grid[v][u]
        return False
    
    def heuristic(self, v1: int, u1: int, v2: int, u2: int) -> float:
        """启发式函数：欧几里得距离"""
        return math.sqrt((v2 - v1) ** 2 + (u2 - u1) ** 2)
    
    def compute_step_cost(self, v: int, u: int, move_cost: float) -> float:
        """
        计算移动到某格子的实际代价
        
        cost = move_cost + λ * danger_penalty
        """
        danger_penalty = self.cspace.get_danger_penalty(v, u)
        return move_cost + self.lambda_weight * danger_penalty
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* 搜索
        
        Args:
            start: (u, v) 起点像素坐标
            goal: (u, v) 终点像素坐标
        
        Returns:
            路径 [(u, v), ...] 或 None（无解）
        """
        start_u, start_v = start
        goal_u, goal_v = goal
        
        # 检查起点和终点
        if not self.is_valid(start_v, start_u):
            print(f"错误：起点 ({start_u}, {start_v}) 在膨胀后的配置空间中不可通行")
            return None
        if not self.is_valid(goal_v, goal_u):
            print(f"错误：终点 ({goal_u}, {goal_v}) 在膨胀后的配置空间中不可通行")
            return None
        
        # 优先队列：(f_score, counter, v, u)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start_v, start_u))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {}
        g_score[(start_v, start_u)] = 0
        
        f_score: Dict[Tuple[int, int], float] = {}
        f_score[(start_v, start_u)] = self.heuristic(start_v, start_u, goal_v, goal_u)
        
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
                
                # 对角线穿墙检查
                if dv != 0 and du != 0:
                    if not self.is_valid(current_v + dv, current_u) or \
                       not self.is_valid(current_v, current_u + du):
                        continue
                
                # 计算代价（含安全惩罚）
                step_cost = self.compute_step_cost(neighbor_v, neighbor_u, move_cost)
                tentative_g = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor_v, neighbor_u, goal_v, goal_u)
                    f_score[neighbor] = f
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor_v, neighbor_u))
        
        print(f"  A* 搜索失败：未找到路径，迭代 {iterations} 次")
        return None
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径，返回 [(u, v), ...] 格式"""
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
# 路径后处理
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
    yaw_rad = math.atan2(dy, dx)
    yaw_deg = math.degrees(yaw_rad)
    return yaw_deg


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
    
    # 创建 RGB 图像（配置空间背景）
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    for v in range(height):
        for u in range(width):
            if not cspace.original_grid[v][u]:
                pixels[u, v] = (30, 30, 30)  # 原始障碍：深灰
            elif not cspace.inflated_grid[v][u]:
                pixels[u, v] = (100, 40, 40)  # 膨胀区域：暗红
            else:
                dist = cspace.distance_field[v][u]
                if dist < cspace.danger_pixels:
                    ratio = dist / cspace.danger_pixels
                    r = int(180 * (1 - ratio) + 50)
                    g = int(150 * ratio + 50)
                    b = 50
                    pixels[u, v] = (r, g, b)
                else:
                    pixels[u, v] = (220, 220, 220)  # 安全区：浅灰
    
    draw = ImageDraw.Draw(img)
    
    # 绘制路径线（蓝色）
    if len(path) > 1:
        for i in range(len(path) - 1):
            draw.line([path[i], path[i+1]], fill=(0, 150, 255), width=3)
    
    # 绘制路径点
    for p in path:
        r = 2
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=(0, 200, 255))
    
    # 起点（绿色）
    r = 6
    draw.ellipse([start[0]-r, start[1]-r, start[0]+r, start[1]+r], fill=(0, 255, 0))
    
    # 终点（红色）
    draw.ellipse([goal[0]-r, goal[1]-r, goal[0]+r, goal[1]+r], fill=(255, 0, 0))
    
    img.save(output_path)
    print(f"路径可视化已保存至: {output_path}")


# ============================================================
# Waypoint 生成
# ============================================================

def generate_waypoints(path_pixels: List[Tuple[int, int]], 
                       transformer: CoordinateTransformer,
                       resample_interval_cm: float = 200.0) -> List[dict]:
    """从像素路径生成 UE waypoint"""
    world_path = [transformer.pixel_to_world(u, v) for u, v in path_pixels]
    world_path = resample_path(world_path, resample_interval_cm)
    
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


def format_ue_commands(waypoints: List[dict]) -> List[str]:
    """将 waypoints 格式化为 UE 控制指令"""
    commands = []
    for wp in waypoints:
        cmd = f"SetPose:{wp['x']:.2f},{wp['y']:.2f},{wp['z']:.2f},{wp['pitch']:.2f},{wp['yaw']:.2f},{wp['roll']:.2f}"
        commands.append(cmd)
    return commands


def save_waypoints(waypoints: List[dict], output_path: str):
    """保存 waypoints 到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(waypoints, f, indent=2, ensure_ascii=False)
    print(f"Waypoints 已保存至: {output_path}")


# ============================================================
# 路径评估指标
# ============================================================

def evaluate_path(path_pixels: List[Tuple[int, int]], 
                  cspace: ConfigurationSpace,
                  transformer: CoordinateTransformer) -> dict:
    """
    评估路径质量
    
    Returns:
        metrics: 包含各项指标的字典
    """
    metrics = {}
    
    # 1. 路径总长度
    world_path = [transformer.pixel_to_world(u, v) for u, v in path_pixels]
    total_length_cm = 0
    for i in range(1, len(world_path)):
        dx = world_path[i][0] - world_path[i-1][0]
        dy = world_path[i][1] - world_path[i-1][1]
        total_length_cm += math.sqrt(dx**2 + dy**2)
    metrics['path_length_m'] = total_length_cm / 100
    
    # 2. 直线距离
    start_world = world_path[0]
    end_world = world_path[-1]
    direct_dist_cm = math.sqrt(
        (end_world[0] - start_world[0])**2 + 
        (end_world[1] - start_world[1])**2
    )
    metrics['direct_distance_m'] = direct_dist_cm / 100
    
    # 3. 路径效率
    metrics['path_efficiency'] = direct_dist_cm / total_length_cm if total_length_cm > 0 else 0
    
    # 4. 最小离障距离（基于原始障碍）
    min_clearance = float('inf')
    for u, v in path_pixels:
        dist = cspace.distance_field[v][u]
        if dist < min_clearance:
            min_clearance = dist
    
    clearance_m = min_clearance * cspace.cm_per_pixel / 100
    metrics['min_clearance_m'] = clearance_m
    
    # 5. 平均离障距离
    total_clearance = sum(cspace.distance_field[v][u] for u, v in path_pixels)
    avg_clearance = total_clearance / len(path_pixels) if path_pixels else 0
    metrics['avg_clearance_m'] = avg_clearance * cspace.cm_per_pixel / 100
    
    # 6. 路径点数
    metrics['path_points'] = len(path_pixels)
    
    return metrics


# ============================================================
# 主函数
# ============================================================

def main():
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(project_dir, 'config')
    result_dir = os.path.join(project_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    config_path = os.path.join(config_dir, 'grid_config.json')
    image_path = os.path.join(result_dir, 'occupancy_grid.png')
    
    # ========== 1. 加载配置和栅格 ==========
    print("=" * 60)
    print("A* 安全路径规划（含配置空间膨胀）")
    print("=" * 60)
    
    config = load_config(config_path)
    print(f"\n[1] 加载配置: {config_path}")
    
    threshold = config['value_convention']['threshold']
    grid = load_occupancy_grid(image_path, threshold)
    print(f"    栅格尺寸: {len(grid[0])} x {len(grid)}")
    
    # 显示规划参数
    planning = config.get('planning', {})
    print(f"\n    规划参数:")
    print(f"      - 膨胀半径: {planning.get('inflation_radius_m', 1.0)} m")
    print(f"      - 安全边距: {planning.get('safety_margin_m', 0.5)} m")
    print(f"      - 安全代价权重 λ: {planning.get('distance_cost_weight', 0.3)}")
    print(f"      - 危险区域半径: {planning.get('danger_zone_m', 3.0)} m")
    
    # ========== 2. 构建配置空间 ==========
    print(f"\n[2] 构建配置空间...")
    cspace = ConfigurationSpace(grid, config)
    
    print("    膨胀障碍物...")
    cspace.inflate_obstacles()
    
    print("    计算距离场...")
    cspace.compute_distance_field()
    
    # 保存配置空间可视化
    cspace_vis_path = os.path.join(result_dir, 'cspace_visualization.png')
    cspace.visualize_cspace(cspace_vis_path)
    
    # ========== 3. 创建坐标转换器 ==========
    transformer = CoordinateTransformer(config)
    
    # ========== 4. 设置起终点 ==========
    test_points = config.get('test_points', {})
    start_pixel = tuple(test_points.get('start_pixel', [200, 10]))
    goal_pixel = tuple(test_points.get('end_pixel', [190, 220]))
    
    print(f"\n[3] 起终点设置:")
    print(f"    起点像素: {start_pixel}")
    print(f"    终点像素: {goal_pixel}")
    
    # ========== 5. A* 搜索 ==========
    print(f"\n[4] 执行 A* 搜索（安全代价模式）...")
    planner = SafeAStarPlanner(cspace)
    path_pixels = planner.search(start_pixel, goal_pixel)
    
    if path_pixels is None:
        print("\n未找到有效路径！")
        print("可能原因：")
        print("  1. 起点或终点被膨胀区域覆盖")
        print("  2. 膨胀后通道被堵死")
        print("建议：减小 inflation_radius_m 或 safety_margin_m")
        return
    
    # ========== 6. 路径后处理 ==========
    print(f"\n[5] 路径后处理...")
    print(f"    原始路径点数: {len(path_pixels)}")
    
    path_simplified = smooth_path_rdp(path_pixels, epsilon=2.0)
    print(f"    RDP 简化后: {len(path_simplified)} 点")
    
    # ========== 7. 生成 waypoints ==========
    waypoints = generate_waypoints(path_simplified, transformer, resample_interval_cm=200.0)
    print(f"    重采样后 waypoints: {len(waypoints)} 个")
    
    # ========== 8. 保存结果 ==========
    print(f"\n[6] 保存结果...")
    
    # 路径可视化
    output_vis_path = os.path.join(result_dir, 'path_result_safe.png')
    visualize_path_on_cspace(cspace, path_pixels, output_vis_path, start_pixel, goal_pixel)

    # Waypoints
    waypoints_path = os.path.join(result_dir, 'waypoints.json')
    save_waypoints(waypoints, waypoints_path)

    # UE 指令
    commands = format_ue_commands(waypoints)
    commands_path = os.path.join(result_dir, 'ue_commands.txt')
    with open(commands_path, 'w') as f:
        f.write('\n'.join(commands))
    print(f"UE 指令已保存至: {commands_path}")
    
    # ========== 9. 路径评估 ==========
    print(f"\n[7] 路径评估指标:")
    print("-" * 40)
    
    metrics = evaluate_path(path_pixels, cspace, transformer)
    
    print(f"    路径总长度:     {metrics['path_length_m']:.2f} m")
    print(f"    直线距离:       {metrics['direct_distance_m']:.2f} m")
    print(f"    路径效率:       {metrics['path_efficiency']*100:.1f}%")
    print(f"    最小离障距离:   {metrics['min_clearance_m']:.2f} m")
    print(f"    平均离障距离:   {metrics['avg_clearance_m']:.2f} m")
    print(f"    路径点数:       {metrics['path_points']}")
    
    # ========== 10. 指令预览 ==========
    print(f"\n[8] UE 指令预览（前5条）:")
    for cmd in commands[:5]:
        print(f"    {cmd}")
    if len(commands) > 5:
        print(f"    ... (共 {len(commands)} 条)")
    
    print("\n" + "=" * 60)
    print("规划完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
