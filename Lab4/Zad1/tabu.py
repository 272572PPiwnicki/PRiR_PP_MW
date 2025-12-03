import time, math, random, os
from concurrent.futures import ProcessPoolExecutor

def generate_random_tsp(num_cities):
    random.seed(23)
    return [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(num_cities)]

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_total_route_length(route, cities):
    return sum(calculate_distance(cities[route[i]], cities[route[(i + 1) % len(route)]]) for i in range(len(route)))

def find_best_neighbor_in_slice(current_route, cities, i_start, i_end):
    best_move, best_cost = None, float('inf')
    num_cities = len(current_route)

    for i in range(i_start, i_end):
        for j in range(i + 1, num_cities):
            route = current_route[:]
            route[i], route[j] = route[j], route[i]
            cost = get_total_route_length(route, cities)
            if cost < best_cost:
                best_cost, best_move = cost, (i, j)
    return best_move, best_cost

def run_tabu_search(num_cities, max_iters, tabu_len, num_workers):
    cities = generate_random_tsp(num_cities)
    curr_route = list(range(num_cities))
    random.seed(42); random.shuffle(curr_route)
    best_cost = get_total_route_length(curr_route, cities)
    tabu_list = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max_iters):
            chunk = num_cities // num_workers
            futures = []
            for w in range(num_workers):
                start = w * chunk
                end = (w + 1) * chunk if w < num_workers - 1 else num_cities - 1
                if start < end:
                    futures.append(executor.submit(find_best_neighbor_in_slice, curr_route, cities, start, end))

            candidates = [f.result() for f in futures if f.result()[0]]
            iter_move, iter_cost = None, float('inf')

            for move, cost in candidates:
                if (move not in tabu_list and move[::-1] not in tabu_list) or cost < best_cost:
                    if cost < iter_cost:
                        iter_cost, iter_move = cost, move

            if iter_move:
                i, j = iter_move
                curr_route[i], curr_route[j] = curr_route[j], curr_route[i]
                tabu_list.append(iter_move)
                if len(tabu_list) > tabu_len: tabu_list.pop(0)
                if iter_cost < best_cost: best_cost = iter_cost

    return best_cost

if __name__ == "__main__":
    N, ITERS, TABU = 50, 200, 20
    print(f"Parametry: N={N}, Iteracje={ITERS}")

    t0 = time.time()
    res_seq = run_tabu_search(N, ITERS, TABU, 1)
    t1 = time.time()
    print(f"Sekwencyjny (1 cpu): {t1-t0:.4f}s")

    t2 = time.time()
    res_par = run_tabu_search(N, ITERS, TABU, 4)
    t3 = time.time()
    print(f"Równoległy ({os.cpu_count()} cpu): {t3-t2:.4f}s")
  
