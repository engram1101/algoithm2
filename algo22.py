import math
import sys
import gzip
import time 
import matplotlib.pyplot as plt
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
                coords.append((city_id, x, y))
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            
    return coords
def get_coords_map(coords):
    return {city_id: (x, y) for city_id, x, y in coords}

def calculate_distance_on_the_fly(city1_id, city2_id, coords_map):
    try:
        x1, y1 = coords_map[city1_id]
        x2, y2 = coords_map[city2_id]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except KeyError:
        return sys.float_info.max
def prim_mst_optimized(coords, coords_map):
    if not coords: return {}
    city_ids = {c[0] for c in coords}
    key = {city_id: sys.float_info.max for city_id in city_ids}
    parent = {city_id: None for city_id in city_ids}
    mst_set = {city_id: False for city_id in city_ids}
    start_node = coords[0][0]
    key[start_node] = 0
    parent[start_node] = -1 
    num_cities = len(coords)
    for i in range(num_cities):
        if i > 0 and i % 100 == 0:
            print(f"MST 생성 진행 중... {i}/{num_cities}", end='\r')
        min_key, u = min(((key[v_id], v_id) for v_id in city_ids if not mst_set[v_id]), default=(sys.float_info.max, -1))
        if u == -1: break
        mst_set[u] = True
        for v_id in city_ids:
            if not mst_set[v_id]:
                dist = calculate_distance_on_the_fly(u, v_id, coords_map)
                if dist < key[v_id]: key[v_id], parent[v_id] = dist, u
    print(f"MST 생성 진행 중... {num_cities}/{num_cities}")
    mst_adj_list = {city_id: [] for city_id in city_ids}
    for city_id, p in parent.items():
        if p is not None and p != -1:
            mst_adj_list[p].append(city_id)
            mst_adj_list[city_id].append(p)
    return mst_adj_list

def pre_order_traversal(mst_adj_list, start_node):
    path, visited, stack = [], set(), [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            path.append(node)
            visited.add(node)
            if node in mst_adj_list:
                stack.extend(reversed(mst_adj_list[node]))
    return path

def create_initial_tour(pre_order_path):
    tour, visited_in_tour = [], set()
    for city in pre_order_path:
        if city not in visited_in_tour:
            tour.append(city)
            visited_in_tour.add(city)
    if tour: tour.append(tour[0])
    return tour
def calculate_tour_cost(tour, coords_map):
    return sum(calculate_distance_on_the_fly(tour[i], tour[i+1], coords_map) for i in range(len(tour)-1))

def two_opt_swap(tour, i, j):
    new_tour = tour[:i+1]
    new_tour.extend(reversed(tour[i+1:j+1]))
    new_tour.extend(tour[j+1:])
    return new_tour

def apply_2_opt(initial_tour, coords_map):
    best_tour = initial_tour
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        print(f"2-Opt 개선 진행 중... (반복 {iteration})")
        improved = False
        best_cost = calculate_tour_cost(best_tour, coords_map)
        
        for i in range(len(best_tour) - 2):
            for j in range(i + 2, len(best_tour) - 1):
                current_dist = calculate_distance_on_the_fly(best_tour[i], best_tour[i+1], coords_map) + \
                               calculate_distance_on_the_fly(best_tour[j], best_tour[j+1], coords_map)
                new_dist = calculate_distance_on_the_fly(best_tour[i], best_tour[j], coords_map) + \
                           calculate_distance_on_the_fly(best_tour[i+1], best_tour[j+1], coords_map)
                
                if new_dist < current_dist:
                    best_tour = two_opt_swap(best_tour, i, j)
                    improved = True
                    break
            if improved:
                break
    return best_tour
def visualize_final_tour(coords, tour, title):
    """최종 경로를 시각화합니다."""
    coords_map = get_coords_map(coords)
    plt.figure(figsize=(12, 12))
    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]
    plt.scatter(x_coords, y_coords, c='black', s=10, zorder=3, label='Cities')

    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        x1, y1 = coords_map[u]
        x2, y2 = coords_map[v]
        plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.8, linewidth=1.2, label='Final Tour' if i == 0 else "")

    plt.title(title, fontsize=16)
    plt.legend()
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
if __name__ == '__main__':
    file_to_test = 'kz9976.tsp' 

    print(f"'{file_to_test}' 파일로 [MST + 2-Opt] 알고리즘 테스트를 시작합니다.")
    print("-" * 50)
    total_start_time = time.time()

    tsp_data_string = read_tsp_file(file_to_test)
    if tsp_data_string is None: sys.exit(1)

    coords = parse_tsp_data(tsp_data_string)
    if not coords:
        print("오류: 도시 데이터를 파싱하지 못했습니다.")
        sys.exit(1)
        
    num_cities = len(coords)
    print(f"{num_cities}개의 도시 데이터를 파싱했습니다.")
    print("-" * 50)
    
    coords_map = get_coords_map(coords)
    print("[단계 1] MST 기반 알고리즘으로 초기 경로를 생성합니다.")
    mst = prim_mst_optimized(coords, coords_map)
    pre_order_path = pre_order_traversal(mst, coords[0][0])
    initial_tour = create_initial_tour(pre_order_path)
    initial_cost = calculate_tour_cost(initial_tour, coords_map)
    print(f"초기 경로 생성 완료. 비용: {initial_cost:.2f}")
    print("-" * 50)

    print("[단계 2] 2-Opt 알고리즘으로 경로를 개선합니다.")
    final_tour = apply_2_opt(initial_tour, coords_map)
    final_cost = calculate_tour_cost(final_tour, coords_map)
    print(f"경로 개선 완료.")
    print("-" * 50)
    
    print("[최종 결과]")
    print(f"초기 경로 비용 (MST-Approx): {initial_cost:.2f}")
    print(f"개선된 경로 비용 (2-Opt):   {final_cost:.2f}")
    improvement = ((initial_cost - final_cost) / initial_cost) * 100
    print(f"개선율: {improvement:.2f}%")
    
    total_end_time = time.time()
    print(f"총 실행 시간: {total_end_time - total_start_time:.2f} 초")
    print("-" * 50)

    print("최종 결과를 시각화합니다...")
    visualize_final_tour(coords, final_tour, f"Final Tour for {file_to_test} (MST + 2-Opt)")