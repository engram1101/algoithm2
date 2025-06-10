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
                coords.append((city_id, x, y))
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            
    return coords

def get_coords_map(coords):
    return {city_id: (x, y) for city_id, x, y in coords}

def calculate_distance(city1_id, city2_id, coords_map):
    try:
        x1, y1 = coords_map[city1_id]
        x2, y2 = coords_map[city2_id]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except KeyError:
        return sys.float_info.max

def calculate_tour_cost(tour, coords_map):
    return sum(calculate_distance(tour[i], tour[i+1], coords_map) for i in range(len(tour)-1))

def simple_grid_cluster(coords, grid_size=2):
    if not coords: return {}
    
    min_x = min(c[1] for c in coords)
    max_x = max(c[1] for c in coords)
    min_y = min(c[2] for c in coords)
    max_y = max(c[2] for c in coords)
    epsilon = 1e-5
    range_x = (max_x - min_x) + epsilon
    range_y = (max_y - min_y) + epsilon
    
    cell_width = range_x / grid_size
    cell_height = range_y / grid_size
    
    clusters = {i: [] for i in range(grid_size * grid_size)}
    
    for city in coords:
        city_id, x, y = city
        col = int((x - min_x) / cell_width)
        row = int((y - min_y) / cell_height)
        cluster_index = row * grid_size + col
        clusters[cluster_index].append(city)
        
    return clusters

def local_tsp_solver(cluster_coords, coords_map):
    if len(cluster_coords) < 2:
        if len(cluster_coords) == 1:
             return [cluster_coords[0][0]]
        return []

    mst = prim_mst_for_cluster(cluster_coords, coords_map)
    pre_order_path = pre_order_traversal_for_cluster(mst, cluster_coords[0][0])
    initial_tour = create_initial_tour_for_cluster(pre_order_path)
    final_tour = apply_2_opt_for_cluster(initial_tour, coords_map)
    
    return final_tour[:-1] 

def prim_mst_for_cluster(coords, coords_map):
    if not coords: return {}
    city_ids = {c[0] for c in coords}
    key = {city_id: sys.float_info.max for city_id in city_ids}
    parent = {city_id: None for city_id in city_ids}
    mst_set = {city_id: False for city_id in city_ids}
    start_node = coords[0][0]
    key[start_node] = 0
    parent[start_node] = -1 
    num_cities = len(coords)
    for _ in range(num_cities):
        min_key, u = min(((key[v_id], v_id) for v_id in city_ids if not mst_set[v_id]), default=(sys.float_info.max, -1))
        if u == -1: break
        mst_set[u] = True
        for v_id in city_ids:
            if not mst_set[v_id]:
                dist = calculate_distance(u, v_id, coords_map)
                if dist < key[v_id]: key[v_id], parent[v_id] = dist, u
    mst_adj_list = {city_id: [] for city_id in city_ids}
    for city_id, p in parent.items():
        if p is not None and p != -1:
            mst_adj_list[p].append(city_id)
            mst_adj_list[city_id].append(p)
    return mst_adj_list

def pre_order_traversal_for_cluster(mst_adj_list, start_node):
    path, visited, stack = [], set(), [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            path.append(node)
            visited.add(node)
            if node in mst_adj_list:
                stack.extend(reversed(mst_adj_list[node]))
    return path

def create_initial_tour_for_cluster(pre_order_path):
    tour, visited_in_tour = [], set()
    for city in pre_order_path:
        if city not in visited_in_tour:
            tour.append(city)
            visited_in_tour.add(city)
    if tour: tour.append(tour[0])
    return tour

def apply_2_opt_for_cluster(initial_tour, coords_map):
    if len(initial_tour) < 4: return initial_tour
    best_tour, improved = initial_tour, True
    while improved:
        improved = False
        for i in range(len(best_tour) - 2):
            for j in range(i + 2, len(best_tour) - 1):
                current_dist = calculate_distance(best_tour[i], best_tour[i+1], coords_map) + calculate_distance(best_tour[j], best_tour[j+1], coords_map)
                new_dist = calculate_distance(best_tour[i], best_tour[j], coords_map) + calculate_distance(best_tour[i+1], best_tour[j+1], coords_map)
                if new_dist < current_dist:
                    best_tour = best_tour[:i+1] + best_tour[i+1:j+1][::-1] + best_tour[j+1:]
                    improved = True
                    break
            if improved: break
    return best_tour
def merge_tours(main_tour, sub_tour, coords_map):
    if not sub_tour: return main_tour
    if not main_tour: return sub_tour
    
    best_insert_cost = sys.float_info.max
    best_new_tour = []
    for i in range(len(main_tour)):
        main_p1 = main_tour[i]
        main_p2 = main_tour[(i + 1) % len(main_tour)]
        for j in range(len(sub_tour)):
            sub_p1 = sub_tour[j]
            original_cost = calculate_distance(main_p1, main_p2, coords_map)
            reordered_sub = sub_tour[j:] + sub_tour[:j]
            sub_p_last = reordered_sub[-1]
            new_cost = calculate_distance(main_p1, sub_p1, coords_map) + calculate_distance(sub_p_last, main_p2, coords_map)
            cost_increase = new_cost - original_cost
            if cost_increase < best_insert_cost:
                best_insert_cost = cost_increase
                new_tour = main_tour[:i+1] + reordered_sub + main_tour[i+1:]
                best_new_tour = new_tour

    return best_new_tour
def visualize_hybrid_solution(coords, clusters, sub_tours, final_tour, title):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    colors = plt.cm.get_cmap('tab10', len(clusters))
    for i, (cluster_id, cities) in enumerate(clusters.items()):
        if not cities: continue
        color = colors(i)
        x_coords = [c[1] for c in cities]
        y_coords = [c[2] for c in cities]
        ax.scatter(x_coords, y_coords, color=color, s=15, label=f'Cluster {cluster_id}')
        
        sub_tour = sub_tours.get(cluster_id, [])
        if sub_tour:
            sub_tour_closed = sub_tour + [sub_tour[0]]
            for k in range(len(sub_tour_closed) - 1):
                u, v = sub_tour_closed[k], sub_tour_closed[k+1]
                x1, y1 = coords_map[u]
                x2, y2 = coords_map[v]
                ax.plot([x1, x2], [y1, y2], color=color, linestyle='--', linewidth=1)
    if final_tour:
        final_tour_closed = final_tour + [final_tour[0]]
        for i in range(len(final_tour_closed) - 1):
            u, v = final_tour_closed[i], final_tour_closed[i+1]
            x1, y1 = coords_map[u]
            x2, y2 = coords_map[v]
            ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.8, linewidth=2, label='Final Merged Tour' if i == 0 else "")

    plt.title(title, fontsize=16)
    plt.legend()
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    ax.set_aspect('equal', adjustable='box')
    plt.show()
if __name__ == '__main__':
    file_to_test = 'kz9976.tsp' 
    GRID_SIZE = 2 

    print(f"'{file_to_test}' 파일로 [클러스터링 기반 하이브리드] 알고리즘 테스트를 시작합니다.")
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
    print(f"[단계 1] 도시들을 {GRID_SIZE}x{GRID_SIZE} 격자로 분할합니다...")
    clusters = simple_grid_cluster(coords, GRID_SIZE)
    print("분할 완료.")
    print("-" * 50)

    print("[단계 2] 각 클러스터 내부에서 지역 최적 경로(서브 투어)를 찾습니다...")
    sub_tours = {}
    for cluster_id, cluster_coords in clusters.items():
        if cluster_coords:
            print(f"  - 클러스터 {cluster_id} (도시 {len(cluster_coords)}개) 처리 중...")
            sub_tours[cluster_id] = local_tsp_solver(cluster_coords, coords_map)
    print("모든 서브 투어 생성 완료.")
    print("-" * 50)

    print("[단계 3] 모든 서브 투어를 하나의 경로로 결합합니다...")
    valid_sub_tours = [st for st in sub_tours.values() if st]
    
    final_tour = []
    if valid_sub_tours:
        final_tour = valid_sub_tours[0]
        for i in range(1, len(valid_sub_tours)):
            print(f"  - 서브 투어 {i+1}/{len(valid_sub_tours)} 결합 중...")
            final_tour = merge_tours(final_tour, valid_sub_tours[i], coords_map)
    
    print("결합 완료.")
    print("-" * 50)
    final_cost = calculate_tour_cost(final_tour + [final_tour[0]] if final_tour else [], coords_map)
    print("[최종 결과]")
    print(f"계산된 경로 비용: {final_cost:.2f}")
    total_end_time = time.time()
    print(f"총 실행 시간: {total_end_time - total_start_time:.2f} 초")
    print("-" * 50)
    print("결과를 시각화합니다...")
    visualize_hybrid_solution(coords, clusters, sub_tours, final_tour, f"Hybrid Solution for {file_to_test}")

