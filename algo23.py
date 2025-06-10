import math
import sys
import gzip
import time 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def read_tsp_file(file_path):
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def parse_tsp_data(data_string):
    coords = []
    in_coord_section = False
    lines = data_string.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if "EOF" in line or "TOUR_SECTION" in line:
            break
        if in_coord_section:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                city_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords.append({'id': city_id, 'x': x, 'y': y})
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            
    return coords

def get_coords_map(coords):
    return {c['id']: (c['x'], c['y']) for c in coords}

def calculate_tour_cost(tour, coords_map):
    if not tour or len(tour) < 2:
        return 0.0
    closed_tour = tour + [tour[0]]
    
    return sum(calculate_distance(closed_tour[i], closed_tour[i+1], coords_map) for i in range(len(closed_tour)-1))

def calculate_distance(city1_id, city2_id, coords_map):
    try:
        x1, y1 = coords_map[city1_id]
        x2, y2 = coords_map[city2_id]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except KeyError:
        return sys.float_info.max

def grid_traversal_heuristic(coords, grid_size):
    if not coords:
        return []
    min_x = min(c['x'] for c in coords)
    max_x = max(c['x'] for c in coords)
    min_y = min(c['y'] for c in coords)
    max_y = max(c['y'] for c in coords)

    epsilon = 1e-5
    range_x = (max_x - min_x) + epsilon
    range_y = (max_y - min_y) + epsilon
    
    cell_width = range_x / grid_size
    cell_height = range_y / grid_size
    grid = [[] for _ in range(grid_size * grid_size)]
    
    for city in coords:
        col = int((city['x'] - min_x) / cell_width)
        row = int((city['y'] - min_y) / cell_height)
        cell_index = row * grid_size + col
        grid[cell_index].append(city['id'])
    tour = []
    for cell_index in range(grid_size * grid_size):
        tour.extend(grid[cell_index])
        
    return tour

def visualize_grid_solution(coords, tour, grid_params, title):
    """격자와 최종 경로를 시각화합니다."""
    coords_map = get_coords_map(coords)
    grid_size, min_x, min_y, cell_width, cell_height = grid_params
    
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    for i in range(grid_size + 1):
        x = min_x + i * cell_width
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.5)
        y = min_y + i * cell_height
        ax.axhline(y, color='gray', linestyle=':', linewidth=0.5)
    x_coords = [c['x'] for c in coords]
    y_coords = [c['y'] for c in coords]
    ax.scatter(x_coords, y_coords, c='black', s=10, zorder=3, label='Cities')
    if tour:
        closed_tour = tour + [tour[0]]
        for i in range(len(closed_tour) - 1):
            u, v = closed_tour[i], closed_tour[i+1]
            x1, y1 = coords_map[u]
            x2, y2 = coords_map[v]
            ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.8, linewidth=1.2, label='Final Tour' if i == 0 else "")

    plt.title(title, fontsize=16)
    plt.legend()
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    ax.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    file_to_test = 'a280.tsp.gz' 

    print(f"'{file_to_test}' 파일로 [격자 순회 휴리스틱] 알고리즘 테스트를 시작합니다.")
    print("-" * 50)
    total_start_time = time.time()

    tsp_data_string = read_tsp_file(file_to_test)
    if tsp_data_string is None: sys.exit(1)

    coords = parse_tsp_data(tsp_data_string)
    if not coords:
        print("오류: 도시 데이터를 파싱하지 못했습니다.")
        sys.exit(1)
    num_cities = len(coords)
    coords_map = get_coords_map(coords)
    print(f"{num_cities}개의 도시 데이터를 파싱했습니다.")
    print("-" * 50)
    grid_size = int(math.sqrt(num_cities / 2)) 
    print(f"알고리즘 실행 시작... ({grid_size}x{grid_size} 격자 사용)")
    final_tour = grid_traversal_heuristic(coords, grid_size)
    final_cost = calculate_tour_cost(final_tour, coords_map)
    print("알고리즘 실행 완료.")
    print("-" * 50)
    print("[최종 결과]")
    print(f"계산된 경로 비용: {final_cost:.2f}")
    total_end_time = time.time()
    print(f"총 실행 시간: {total_end_time - total_start_time:.2f} 초")
    print("-" * 50)
    min_x = min(c['x'] for c in coords)
    max_x = max(c['x'] for c in coords)
    min_y = min(c['y'] for c in coords)
    max_y = max(c['y'] for c in coords)
    epsilon = 1e-5
    cell_width = ((max_x - min_x) + epsilon) / grid_size
    cell_height = ((max_y - min_y) + epsilon) / grid_size
    grid_params = (grid_size, min_x, min_y, cell_width, cell_height)

    print("최종 결과를 시각화합니다...")
    if num_cities > 10000:
        print("주의: 도시 개수가 매우 많아 시각화에 시간이 오래 걸릴 수 있습니다.")
    visualize_grid_solution(coords, final_tour, grid_params, f"Grid Traversal Heuristic for {file_to_test}")

