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
    if not coords:
        return {}
    
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

        min_key = sys.float_info.max
        u = -1
        for v_id in city_ids:
            if not mst_set[v_id] and key[v_id] < min_key:
                min_key = key[v_id]
                u = v_id
        
        if u == -1: break
        mst_set[u] = True
        
        for v_id in city_ids:
            if not mst_set[v_id]:
                dist = calculate_distance_on_the_fly(u, v_id, coords_map)
                if dist < key[v_id]:
                    key[v_id] = dist
                    parent[v_id] = u
    
    print(f"MST 생성 진행 중... {num_cities}/{num_cities}") 

    mst_adj_list = {city_id: [] for city_id in city_ids}
    for city_id in city_ids:
        p = parent.get(city_id)
        if p is not None and p != -1:
            mst_adj_list[p].append(city_id)
            mst_adj_list[city_id].append(p)
            
    return mst_adj_list

def pre_order_traversal(mst_adj_list, start_node):
    path = []
    visited = set()
    
    def dfs(node):
        path.append(node)
        visited.add(node)
        if node in mst_adj_list:
            for neighbor in mst_adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor)
    
    if mst_adj_list:
        dfs(start_node)
    return path

def create_tour_from_pre_order(pre_order_path):
    tour = []
    visited_in_tour = set()
    for city in pre_order_path:
        if city not in visited_in_tour:
            tour.append(city)
            visited_in_tour.add(city)
    if tour:
        tour.append(tour[0])
    return tour

def calculate_tour_cost(tour, coords_map):
    total_cost = 0
    for i in range(len(tour) - 1):
        city1 = tour[i]
        city2 = tour[i+1]
        total_cost += calculate_distance_on_the_fly(city1, city2, coords_map)
    return total_cost
def visualize_solution(coords, mst, tour, title):
    coords_map = get_coords_map(coords)
    plt.figure(figsize=(10, 10))

    if len(coords) <= 5000:
        x_coords = [c[1] for c in coords]
        y_coords = [c[2] for c in coords]
        plt.scatter(x_coords, y_coords, c='black', s=5, zorder=3, label='Cities')

    drawn_edges = set()
    for u, neighbors in mst.items():
        for v in neighbors:
            edge = tuple(sorted((u, v)))
            if edge not in drawn_edges:
                x1, y1 = coords_map[u]
                x2, y2 = coords_map[v]
                plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.5, linewidth=0.8, label='MST' if not drawn_edges else "")
                drawn_edges.add(edge)

    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        x1, y1 = coords_map[u]
        x2, y2 = coords_map[v]
        plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.7, linewidth=1, label='TSP Tour' if i == 0 else "")

    plt.title(title, fontsize=16)
    plt.legend()
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    file_to_test = 'mona-lisa100K.tsp' 

    print(f"'{file_to_test}' 파일로 테스트를 시작합니다.")
    print("-" * 40)
    start_time = time.time()

    tsp_data_string = read_tsp_file(file_to_test)
    if tsp_data_string is None: sys.exit(1)

    coords = parse_tsp_data(tsp_data_string)
    if not coords:
        print("데이터에서 도시 좌표를 파싱할 수 없습니다.")
        sys.exit(1)
        
    num_cities = len(coords)
    print(f"{num_cities}개의 도시 데이터를 파싱했습니다.")
    
    if num_cities + 10 > sys.getrecursionlimit():
        sys.setrecursionlimit(num_cities + 10)

    print("-" * 40)
    
    coords_map = get_coords_map(coords)

    print("MST 생성을 시작합니다...")
    mst = prim_mst_optimized(coords, coords_map)
    print("MST 생성 완료.")
    print("-" * 40)
    
    print("경로를 생성하고 비용을 계산합니다...")
    pre_order_path = pre_order_traversal(mst, coords[0][0])
    final_tour = create_tour_from_pre_order(pre_order_path)
    tour_cost = calculate_tour_cost(final_tour, coords_map)
    
    end_time = time.time()
    
    print(f"최종 TSP 경로 생성 완료 (총 {len(final_tour)-1}개 도시 방문).")
    print(f"계산된 총 경로 비용: {tour_cost:.2f}")
    print(f"총 실행 시간: {end_time - start_time:.2f} 초")
    print("-" * 40)

    print("결과를 시각화합니다... (그래프 창을 닫으면 프로그램이 종료됩니다)")
    print("주의: 도시 개수가 많으면 시각화에 시간이 오래 걸리거나 프로그램이 응답하지 않을 수 있습니다.")
    visualize_solution(coords, mst, final_tour, f"TSP Solution for {file_to_test}")
