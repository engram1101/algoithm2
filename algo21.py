import math
import sys
import gzip
import time
import itertools 
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
                original_id = int(parts[0])
                city_id = original_id - 1 
                x = float(parts[1])
                y = float(parts[2])
                coords.append((city_id, x, y, original_id)) 
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            
    return coords

def calculate_distance_matrix(coords):
    num_cities = len(coords)
    distance_matrix = [[0.0] * num_cities for _ in range(num_cities)]
    
    for i in range(num_cities):
        for j in range(i, num_cities):
            _, x1, y1, _ = coords[i]
            _, x2, y2, _ = coords[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
            
    return distance_matrix


def held_karp(dist_matrix):
    n = len(dist_matrix)
    C = [[sys.float_info.max] * n for _ in range(1 << n)]
    C[1][0] = 0

    for s in range(2, n + 1):
        print(f"부분 집합 크기 {s}/{n} 처리 중...")
        for comb in itertools.combinations(range(n), s):
            if 0 not in comb:
                continue
            mask = 0
            for i in comb:
                mask |= (1 << i)
            for j in comb:
                if j == 0:
                    continue
                prev_mask = mask ^ (1 << j)
                min_dist = sys.float_info.max
                for k in comb:
                    if k == j:
                        continue
                    dist = C[prev_mask][k] + dist_matrix[k][j]
                    if dist < min_dist:
                        min_dist = dist
                C[mask][j] = min_dist
    full_mask = (1 << n) - 1
    min_tour_cost = sys.float_info.max
    last_city = -1
    for j in range(1, n):
        cost = C[full_mask][j] + dist_matrix[j][0]
        if cost < min_tour_cost:
            min_tour_cost = cost
            last_city = j
    tour = [0] * n
    current_mask = full_mask
    current_city = last_city
    for i in range(n - 1, 0, -1):
        tour[i] = current_city
        prev_mask = current_mask ^ (1 << current_city)
        for prev_city in range(n):
            if (prev_mask >> prev_city) & 1:
                if math.isclose(C[current_mask][current_city], C[prev_mask][prev_city] + dist_matrix[prev_city][current_city]):
                    current_mask = prev_mask
                    current_city = prev_city
                    break
    tour[0] = 0
    return min_tour_cost, tour

def visualize_tour(coords, tour, title):
    
    coords_map = {city_id: (x, y) for city_id, x, y, _ in coords}
    tour_coords = [coords_map[i] for i in tour]
    tour_coords.append(tour_coords[0]) 
    x, y = zip(*tour_coords)
    plt.figure(figsize=(10, 10))
    plt.scatter([c[1] for c in coords], [c[2] for c in coords], c='black', s=25, zorder=3, label='Cities')
    plt.plot(x, y, 'r-', alpha=0.8, linewidth=1.5, label='Optimal Tour')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    file_to_test = 'a280.tsp.gz' 
    print("Held-Karp 알고리즘 테스트를 시작합니다.")
    print(f"'{file_to_test}' 파일의 전체 데이터를 사용합니다.")
    print("-" * 40)
    start_time = time.time()
    tsp_data_string = read_tsp_file(file_to_test)
    if tsp_data_string is None:
        sys.exit(1)
    coords = parse_tsp_data(tsp_data_string)
    num_cities = len(coords)
    if num_cities == 0:
        print("오류: 도시 데이터를 파싱하지 못했습니다. 입력 데이터를 확인해주세요.")
        sys.exit(1)
    print(f"{num_cities}개의 도시 데이터를 사용합니다.")
    
    dist_matrix = calculate_distance_matrix(coords)

    optimal_cost, optimal_tour_0_based = held_karp(dist_matrix)
    
    end_time = time.time()

    original_ids_map = {city_id: original_id for city_id, _, _, original_id in coords}
    optimal_tour_original = [original_ids_map[i] for i in optimal_tour_0_based]
    optimal_tour_original.append(optimal_tour_original[0]) 

    print("-" * 40)
    print("Held-Karp 알고리즘 실행 완료.")
    print(f"최적 경로 비용: {optimal_cost:.2f}")
    print(f"최적 경로 (원래 ID): {optimal_tour_original}")
    print(f"총 실행 시간: {end_time - start_time:.2f} 초")
    print("-" * 40)

    print("결과를 시각화합니다... (그래프 창을 닫으면 프로그램이 종료됩니다)")
    visualize_tour(coords, optimal_tour_0_based, f"Held-Karp Optimal Solution (n={num_cities})")