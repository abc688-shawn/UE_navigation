"""
生成 Occupancy Grid 测试图 (400×240)
白色(255) = 可通行
黑色(0) = 障碍

障碍物设计：
- 中央大型障碍群（需要绕行）
- 窄通道测试
- 分散的小障碍
- 边界墙
"""

import os

from PIL import Image, ImageDraw


def create_occupancy_grid(width=400, height=240):
    """
    创建一个有挑战性的占据栅格图
    
    Returns:
        PIL.Image: 灰度图像
    """
    # 创建白色背景 (255 = free)
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # ========== 1. 边界墙（留出入口）==========
    wall_thickness = 5
    # 上边界（留中间入口）
    draw.rectangle([0, 0, 150, wall_thickness], fill=0)
    draw.rectangle([250, 0, width-1, wall_thickness], fill=0)
    # 下边界（留中间入口）
    draw.rectangle([0, height-wall_thickness-1, 180, height-1], fill=0)
    draw.rectangle([220, height-wall_thickness-1, width-1, height-1], fill=0)
    # 左边界
    draw.rectangle([0, 0, wall_thickness, height-1], fill=0)
    # 右边界
    draw.rectangle([width-wall_thickness-1, 0, width-1, height-1], fill=0)
    
    # ========== 2. 中央大型障碍群（迷宫核心）==========
    # 中央水平长条
    draw.rectangle([80, 100, 320, 130], fill=0)
    
    # 上方垂直挡板（形成通道）
    draw.rectangle([120, 40, 140, 100], fill=0)
    draw.rectangle([200, 30, 220, 100], fill=0)
    draw.rectangle([280, 50, 300, 100], fill=0)
    
    # 下方垂直挡板
    draw.rectangle([100, 130, 120, 190], fill=0)
    draw.rectangle([160, 130, 180, 210], fill=0)
    draw.rectangle([240, 130, 260, 180], fill=0)
    draw.rectangle([310, 130, 330, 200], fill=0)
    
    # ========== 3. 窄通道区域（左侧）==========
    # 交错的水平障碍，形成S形通道
    draw.rectangle([20, 50, 70, 65], fill=0)
    draw.rectangle([40, 90, 90, 105], fill=0)
    draw.rectangle([20, 140, 65, 155], fill=0)
    draw.rectangle([45, 180, 85, 195], fill=0)
    
    # ========== 4. 右侧障碍区 ==========
    # 斜向排列的矩形障碍
    draw.rectangle([340, 40, 370, 70], fill=0)
    draw.rectangle([350, 90, 380, 110], fill=0)
    draw.rectangle([340, 140, 375, 165], fill=0)
    draw.rectangle([355, 185, 385, 210], fill=0)
    
    # ========== 5. 分散的小障碍（增加复杂度）==========
    small_obstacles = [
        # (x1, y1, x2, y2) - 左上角和右下角
        (145, 60, 165, 80),    # 上方通道中的障碍
        (225, 55, 245, 75),
        (265, 65, 285, 85),
        
        (130, 160, 150, 175),  # 下方区域
        (190, 185, 210, 205),
        (270, 155, 290, 175),
        
        # 入口附近的干扰障碍
        (175, 15, 195, 30),    # 上入口
        (195, 210, 215, 225),  # 下入口
    ]
    
    for obs in small_obstacles:
        draw.rectangle(obs, fill=0)
    
    # ========== 6. 额外的挑战元素 ==========
    # 中央通道中的阻挡（必须选择上或下绕行）
    draw.rectangle([185, 105, 215, 125], fill=0)  # 堵住中央缺口
    
    # 创造一个必须穿过的狭窄通道
    draw.rectangle([55, 115, 75, 145], fill=0)
    
    return img


def add_test_points(img, start_pixel=(30, 30), end_pixel=(370, 210)):
    """
    在图像上标记起点和终点（用于可视化验证）
    注意：这会修改图像，正式生成时可以不调用
    """
    draw = ImageDraw.Draw(img)
    # 用灰色圆圈标记（不影响黑白判断）
    r = 5
    # 起点 - 浅灰色
    draw.ellipse([start_pixel[0]-r, start_pixel[1]-r, 
                  start_pixel[0]+r, start_pixel[1]+r], fill=200)
    # 终点 - 浅灰色
    draw.ellipse([end_pixel[0]-r, end_pixel[1]-r,
                  end_pixel[0]+r, end_pixel[1]+r], fill=200)
    return img


def visualize_grid_info(img):
    """打印栅格统计信息"""
    pixels = list(img.getdata())
    total_cells = len(pixels)
    free_cells = sum(1 for p in pixels if p == 255)
    occupied_cells = sum(1 for p in pixels if p == 0)
    other_cells = total_cells - free_cells - occupied_cells
    
    print(f"栅格尺寸: {img.width} x {img.height} = {total_cells} 格子")
    print(f"可通行区域 (白): {free_cells} ({100*free_cells/total_cells:.1f}%)")
    print(f"障碍区域 (黑): {occupied_cells} ({100*occupied_cells/total_cells:.1f}%)")
    if other_cells > 0:
        print(f"其他灰度值: {other_cells}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    result_dir = os.path.join(project_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    # 生成栅格图
    print("正在生成 Occupancy Grid...")
    img = create_occupancy_grid(width=400, height=240)
    
    # 打印统计信息
    visualize_grid_info(img)
    
    # 保存图像
    output_path = os.path.join(result_dir, "occupancy_grid.png")
    img.save(output_path)
    print(f"\n已保存至: {output_path}")
    
    # 可选：生成带标记点的预览版本
    img_preview = img.copy()
    img_preview = add_test_points(img_preview, start_pixel=(30, 30), end_pixel=(370, 210))
    preview_path = os.path.join(result_dir, "occupancy_grid_preview.png")
    img_preview.save(preview_path)
    print(f"预览版本（带起终点标记）: {preview_path}")
    
    print("\n建议的测试起终点（像素坐标）:")
    print("  起点: (30, 30)  - 左上区域")
    print("  终点: (370, 210) - 右下区域")
    print("\n对应的 UE 世界坐标（根据 grid_config.json）:")
    print("  起点: x=-8475cm, y=-5475cm, z=-500cm")
    print("  终点: x=8525cm, y=4525cm, z=-500cm")


if __name__ == "__main__":
    main()
