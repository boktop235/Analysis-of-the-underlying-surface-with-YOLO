import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
import math

def create_obstacles():
    obstacles = []
    obstacles.append({'type': 'building', 'x': 2, 'y': 1, 'w': 3, 'd': 2, 'h': 8, 'color': 'red'})
    obstacles.append({'type': 'building', 'x': 7, 'y': 6, 'w': 3, 'd': 3, 'h': 10, 'color': 'darkred'})
    obstacles.append({'type': 'building', 'x': 4, 'y': 8, 'w': 2, 'd': 2, 'h': 9, 'color': 'red'})
    obstacles.append({'type': 'building', 'x': 0, 'y': 4, 'w': 2, 'd': 2, 'h': 3, 'color': 'brown'})
    obstacles.append({'type': 'building', 'x': 9, 'y': 1, 'w': 2, 'd': 3, 'h': 4, 'color': 'brown'})
    obstacles.append({'type': 'building', 'x': 3, 'y': 5, 'w': 2, 'd': 1.5, 'h': 7, 'color': 'red'})
    obstacles.append({'type': 'building', 'x': 8, 'y': 9, 'w': 2, 'd': 2, 'h': 6, 'color': 'red'})
    trees = [(1, 7), (3, 3), (4, 6), (6, 2), (6, 5), (7, 8), (9, 4), (10, 3), (2, 9), (5, 0), (8, 11)]
    for i, (x, y) in enumerate(trees):
        height = 2.5 + (i % 3) * 0.5
        obstacles.append({'type': 'tree', 'x': x, 'y': y, 'w': 1, 'd': 1, 'h': height, 'color': 'green'})
    poles = [(5, 2), (5, 4), (5, 7), (5, 9), (1, 2), (10, 7)]
    for x, y in poles:
        obstacles.append({'type': 'pole', 'x': x, 'y': y, 'w': 0.5, 'd': 0.5, 'h': 6, 'color': 'gray'})
    obstacles.append({'type': 'fence', 'x': 2, 'y': 5.5, 'w': 5, 'd': 0.3, 'h': 1.5, 'color': 'saddlebrown'})
    obstacles.append({'type': 'water', 'x': 0, 'y': 9.5, 'w': 3, 'd': 2.5, 'h': 0.5, 'color': 'cyan'})
    obstacles.append({'type': 'landing', 'x': 10, 'y': 10, 'w': 2, 'd': 2, 'h': 0, 'color': 'limegreen'})
    return obstacles

def get_height_at(x, y, obstacles):
    height = 0
    for obs in obstacles:
        if obs['x'] <= x < obs['x'] + obs['w'] and obs['y'] <= y < obs['y'] + obs['d']:
            if obs['h'] > height:
                height = obs['h']
    return height

def build_heightmap(width, height, obstacles):
    heightmap = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            heightmap[i, j] = get_height_at(i, j, obstacles)
    return heightmap

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star_fixed(heightmap, start, goal, drone_height):
    width, height = heightmap.shape
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def move_cost(a, b):
        dx = abs(a[0]-b[0])
        dy = abs(a[1]-b[1])
        return math.sqrt(dx*dx + dy*dy)
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    open_set = [(f_score[start], start)]
    closed_set = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        closed_set.add(current)
        for dx, dy in neighbors:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if nx<0 or nx>=width or ny<0 or ny>=height:
                continue
            if neighbor in closed_set:
                continue
            if heightmap[nx, ny] >= drone_height:
                continue
            tentative_g = g_score[current] + move_cost(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def a_star_variable(heightmap, start, goal, start_height, goal_height):
    width, height = heightmap.shape
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def move_cost(a, b):
        dx = abs(a[0]-b[0])
        dy = abs(a[1]-b[1])
        return math.sqrt(dx*dx + dy*dy)
    estimated_steps = max(heuristic(start, goal) * 2, 20)
    def target_height(step):
        t = min(1.0, step / estimated_steps)
        return start_height * (1 - t) + goal_height * t
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    steps = {start: 0}
    open_set = [(f_score[start], start)]
    closed_set = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        closed_set.add(current)
        current_step = steps[current]
        for dx, dy in neighbors:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if nx<0 or nx>=width or ny<0 or ny>=height:
                continue
            if neighbor in closed_set:
                continue
            neighbor_step = current_step + 1
            neighbor_target_h = target_height(neighbor_step)
            safe_height = heightmap[nx, ny] + 0.5
            if safe_height > neighbor_target_h + 2:
                continue
            tentative_g = g_score[current] + move_cost(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                steps[neighbor] = neighbor_step
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def draw_cube(ax, x, y, z, w, d, h, color, alpha=0.6):
    vertices = [
        [(x, y, z), (x+w, y, z), (x+w, y+d, z), (x, y+d, z)],
        [(x, y, z+h), (x+w, y, z+h), (x+w, y+d, z+h), (x, y+d, z+h)],
        [(x, y, z), (x+w, y, z), (x+w, y, z+h), (x, y, z+h)],
        [(x, y+d, z), (x+w, y+d, z), (x+w, y+d, z+h), (x, y+d, z+h)],
        [(x, y, z), (x, y+d, z), (x, y+d, z+h), (x, y, z+h)],
        [(x+w, y, z), (x+w, y+d, z), (x+w, y+d, z+h), (x+w, y, z+h)]
    ]
    faces = Poly3DCollection(vertices, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.3)
    ax.add_collection3d(faces)

def plot_3d(obstacles, path, start, goal, drone_height, title, filename, start_h=None, goal_h=None):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for obs in obstacles:
        draw_cube(ax, obs['x'], obs['y'], 0, obs['w'], obs['d'], obs['h'], obs['color'], alpha=0.6)
    if start_h is not None and goal_h is not None:
        ax.scatter(start[0]+0.5, start[1]+0.5, start_h, color='blue', s=150, edgecolors='black', linewidth=2)
        ax.scatter(goal[0]+0.5, goal[1]+0.5, goal_h, color='gold', s=150, edgecolors='black', linewidth=2)
        if path:
            path_x = [p[0]+0.5 for p in path]
            path_y = [p[1]+0.5 for p in path]
            t = np.linspace(0, 1, len(path))
            path_z = start_h * (1-t) + goal_h * t
            ax.plot(path_x, path_y, path_z, color='blue', linewidth=3, marker='o', markersize=3)
    else:
        ax.scatter(start[0]+0.5, start[1]+0.5, drone_height, color='blue', s=150, edgecolors='black', linewidth=2)
        ax.scatter(goal[0]+0.5, goal[1]+0.5, drone_height, color='gold', s=150, edgecolors='black', linewidth=2)
        if path:
            path_x = [p[0]+0.5 for p in path]
            path_y = [p[1]+0.5 for p in path]
            path_z = [drone_height] * len(path)
            ax.plot(path_x, path_y, path_z, color='blue', linewidth=3, marker='o', markersize=3)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Высота (м)')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, 12)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def plot_2d_top(heightmap, path, start, goal, title, filename, start_h=None, goal_h=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heightmap.T, origin='lower', cmap='terrain', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Высота препятствия (м)')
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            if heightmap[i, j] > 0:
                rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.3, alpha=0.5)
                ax.add_patch(rect)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-o', linewidth=2.5, markersize=4)
    ax.scatter(start[0], start[1], color='blue', s=150, edgecolors='black', zorder=5)
    ax.scatter(goal[0], goal[1], color='gold', s=150, edgecolors='black', zorder=5)
    if start_h is not None and goal_h is not None:
        ax.set_title(f'{title}\nСтарт: высота {start_h} м, Цель: высота {goal_h} м')
    else:
        ax.set_title(title)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def plot_side_view(path, start, goal, start_h, goal_h, obstacles, title, filename):
    fig, ax = plt.subplots(figsize=(14, 6))
    max_heights = {}
    for obs in obstacles:
        for x in range(int(obs['x']), int(obs['x'] + obs['w'])):
            if x not in max_heights:
                max_heights[x] = 0
            max_heights[x] = max(max_heights[x], obs['h'])
    for x, h in max_heights.items():
        if h > 0:
            ax.add_patch(plt.Rectangle((x-0.5, 0), 1, h, color='red', alpha=0.5, edgecolor='black', linewidth=0.5))
    if path:
        path_x = [p[0] + 0.5 for p in path]
        t = np.linspace(0, 1, len(path))
        path_z = start_h * (1-t) + goal_h * t
        ax.plot(path_x, path_z, 'b-o', linewidth=2.5, markersize=4)
    ax.axhline(y=start_h, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=goal_h, color='gold', linestyle='--', alpha=0.5)
    ax.scatter(start[0]+0.5, start_h, color='blue', s=100, edgecolors='black', zorder=5)
    ax.scatter(goal[0]+0.5, goal_h, color='gold', s=100, edgecolors='black', zorder=5)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Высота (м)')
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 12)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

MAP_SIZE = 12
obstacles = create_obstacles()
heightmap = build_heightmap(MAP_SIZE, MAP_SIZE, obstacles)

start1 = (0, 0)
goal1 = (11, 11)
height1 = 2.5
path1 = a_star_fixed(heightmap, start1, goal1, height1)
plot_3d(obstacles, path1, start1, goal1, height1, "Эксперимент 1: низкий полет (2.5 м)", "exp1_3d.png")
plot_2d_top(heightmap, path1, start1, goal1, "Эксперимент 1: вид сверху (низкий полет 2.5 м)", "exp1_top.png")

start2 = (0, 5)
goal2 = (11, 5)
height2 = 6.5
path2 = a_star_fixed(heightmap, start2, goal2, height2)
plot_3d(obstacles, path2, start2, goal2, height2, "Эксперимент 2: высокий полет (6.5 м)", "exp2_3d.png")
plot_2d_top(heightmap, path2, start2, goal2, "Эксперимент 2: вид сверху (высокий полет 6.5 м)", "exp2_top.png")

start3 = (0, 10)
goal3 = (10, 0)
height3 = 4.0
path3 = a_star_fixed(heightmap, start3, goal3, height3)
plot_3d(obstacles, path3, start3, goal3, height3, "Эксперимент 3: средняя высота (4.0 м)", "exp3_3d.png")
plot_2d_top(heightmap, path3, start3, goal3, "Эксперимент 3: вид сверху (средняя высота 4.0 м)", "exp3_top.png")

start4 = (1, 0)
goal4 = (11, 9)
start_h4 = 2.0
goal_h4 = 7.0
path4 = a_star_variable(heightmap, start4, goal4, start_h4, goal_h4)
plot_3d(obstacles, path4, start4, goal4, None, "Эксперимент 4: набор высоты 2→7 м", "exp4_3d.png", start_h=start_h4, goal_h=goal_h4)
plot_2d_top(heightmap, path4, start4, goal4, "Эксперимент 4: вид сверху (набор высоты 2→7 м)", "exp4_top.png", start_h=start_h4, goal_h=goal_h4)
plot_side_view(path4, start4, goal4, start_h4, goal_h4, obstacles, "Эксперимент 4: вид сбоку (изменение высоты)", "exp4_side.png")
