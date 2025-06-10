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
                coords.append({'id': city_id, 'x': x, 'y': y})
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            
    return coords

def get_coords_map(coords):
    return {c['id']: (c['x'], c['y']) for c in coords}

def calculate_distance(city1_id, city2_id, coords_map):
    try:
        x1, y1 = coords_map[city1_id]
        x2, y2 = coords_map[city2_id]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except KeyError:
        return sys.float_info.max

def calculate_tour_cost(tour, coords_map):
    return sum(calculate_distance(tour[i], tour[i+1], coords_map) for i in range(len(tour)-1))
def cross_product(p1, p2, p3):
    return (p2['x'] - p1['x']) * (p3['y'] - p1['y']) - (p2['y'] - p1['y']) * (p3['x'] - p1['x'])
def get_convex_hull(coords):
    if len(coords) < 3:
        return [c['id'] for c in coords]
    sorted_coords = sorted(coords, key=lambda p: (p['x'], p['y']))
    upper_hull = []
    for p in sorted_coords:
        while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], p) <= 0:
            upper_hull.pop()
        upper_hull.append(p)
    lower_hull = []
    for p in reversed(sorted_coords):
        while len(lower_hull) >= 2 and cross_product(lower_hull[-2], lower_hull[-1], p) <= 0:
            lower_hull.pop()
        lower_hull.append(p)
    hull_points = upper_hull[:-1] + lower_hull[:-1]
    return [p['id'] for p in hull_points]

def convex_hull_insertion(coords, coords_map):
    num_cities = len(coords)
    print("  - 단계 1: 볼록 껍질(Convex Hull)을 찾습니다...")
    initial_tour = get_convex_hull(coords)
    
    tour = initial_tour
    
    remaining_cities = {c['id'] for c in coords} - set(tour)
    print(f"  - 볼록 껍질에 {len(tour)}개의 도시가 포함되었습니다.")
    print("  - 단계 2: 내부 도시들을 최적의 위치에 삽입합니다...")
    while remaining_cities:
        if (num_cities - len(remaining_cities)) % 100 == 0:
            print(f"    - 경로 생성 진행 중... {num_cities - len(remaining_cities)}/{num_cities}", end='\r')
        farthest_city = -1
        max_min_dist = -1
        for city_k_id in remaining_cities:
            min_dist_to_tour = min(calculate_distance(city_k_id, tour_city_id, coords_map) for tour_city_id in tour)
            if min_dist_to_tour > max_min_dist:
                max_min_dist = min_dist_to_tour
                farthest_city = city_k_id
        
        best_insert_pos = -1
        min_cost_increase = sys.float_info.max
        closed_tour = tour + [tour[0]]
        for i in range(len(closed_tour) - 1):
            city_i_id = closed_tour[i]
            city_j_id = closed_tour[i+1]
            
            cost_increase = calculate_distance(city_i_id, farthest_city, coords_map) + \
                            calculate_distance(farthest_city, city_j_id, coords_map) - \
                            calculate_distance(city_i_id, city_j_id, coords_map)
            
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_insert_pos = i + 1
        tour.insert(best_insert_pos, farthest_city)
        remaining_cities.remove(farthest_city)
    
    print(f"    - 경로 생성 진행 중... {num_cities}/{num_cities}      ")
    tour.append(tour[0])
    return tour

def visualize_solution(coords, hull, final_tour, title):
    """볼록 껍질과 최종 경로를 시각화합니다."""
    coords_map = get_coords_map(coords)
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    x_coords = [c['x'] for c in coords]
    y_coords = [c['y'] for c in coords]
    ax.scatter(x_coords, y_coords, c='black', s=10, zorder=3, label='Cities')
    hull_closed = hull + [hull[0]]
    for i in range(len(hull_closed) - 1):
        u, v = hull_closed[i], hull_closed[i+1]
        x1, y1 = coords_map[u]
        x2, y2 = coords_map[v]
        ax.plot([x1, x2], [y1, y2], 'b--', alpha=0.7, linewidth=1.5, label='Convex Hull' if i == 0 else "")
    for i in range(len(final_tour) - 1):
        u, v = final_tour[i], final_tour[i+1]
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
    file_to_test = 'kz9976.tsp' 

    print(f"'{file_to_test}' 파일로 [볼록 껍질 기반 삽입] 알고리즘 테스트를 시작합니다.")
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
    print("알고리즘 실행 시작...")
    final_tour = convex_hull_insertion(coords, coords_map)
    final_cost = calculate_tour_cost(final_tour, coords_map)
    print("알고리즘 실행 완료.")
    print("-" * 50)
    print("[최종 결과]")
    print(f"계산된 경로 비용: {final_cost:.2f}")
    total_end_time = time.time()
    print(f"총 실행 시간: {total_end_time - total_start_time:.2f} 초")
    print("-" * 50)
    hull_ids = get_convex_hull(coords)
    print("최종 결과를 시각화합니다...")
    visualize_solution(coords, hull_ids, final_tour, f"Convex Hull Insertion for {file_to_test}")

