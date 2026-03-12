import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

INF = math.inf

Edge = Tuple[int, int, int]

def read_input_file_MattKeith(path: str) -> Tuple[int, List[List[int]]]:
    # read adjacency matrix from the input file
    with open(path, "r", encoding="utf-8") as infile:
        lines = [line.strip() for line in infile if line.strip()]

    if not lines:
        raise ValueError(f"Input file is empty: {path}")

    n = int(lines[0])
    if len(lines) != n + 1:
        raise ValueError(
            f"Expected {n} matrix rows after the first line in {path}, found {len(lines) - 1}."
        )

    matrix: List[List[int]] = []
    for row_index, line in enumerate(lines[1:], start=0):
        row = [int(value) for value in line.split()]
        if len(row) != n:
            raise ValueError(
                f"Row {row_index} in {path} has {len(row)} entries; expected {n}."
            )
        matrix.append(row)

    return n, matrix

def matrix_to_edges_MattKeith(matrix: List[List[int]]) -> List[Edge]:
    # convert the matrix into an edge list
    # skip diagonal entries and treat 0 as "no edge"
    n = len(matrix)
    edges: List[Edge] = []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            weight = matrix[u][v]
            if weight != 0:
                edges.append((u, v, weight))
    return edges

def bellman_ford_final_MattKeith(n: int, edges: List[Edge], source: int = 0) -> Tuple[bool, List[float], List[Optional[int]]]:
    # run the standard Bellman‑Ford algorithm
    # returns (negative_cycle_found, dist, predecessor)
    dist: List[float] = [INF] * n
    pred: List[Optional[int]] = [None] * n
    dist[source] = 0

    # Relax all edges V - 1 times.
    for _ in range(n - 1):
        updated = False
        for u, v, weight in edges:
            if dist[u] != INF and dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                pred[v] = u
                updated = True
        if not updated:
            break
    # One additional pass to detect a reachable negative cycle.
    for u, v, weight in edges:
        if dist[u] != INF and dist[v] > dist[u] + weight:
            return True, dist, pred

    return False, dist, pred

def reconstruct_path_MattKeith(pred: List[Optional[int]], source: int, target: int) -> List[int]:
    # rebuild the path from source to target using the pred array
    if target == source:
        return [source]

    path: List[int] = []
    current: Optional[int] = target
    seen = set()

    while current is not None:
        seen.add(current)
        path.append(current)
        if current == source:
            path.reverse()
            return path
        current = pred[current]

    return []

def format_output_MattKeith(n: int, has_negative_cycle: bool, dist: List[float], pred: List[Optional[int]], source: int = 0) -> str:
    # format the output the same way the assignment examples show
    if has_negative_cycle:
        return "There is a negative cycle"

    lines: List[str] = []
    for vertex in range(n):
        if dist[vertex] == INF:
            lines.append(f"{vertex}, inf, []")
            continue
        path = reconstruct_path_MattKeith(pred, source, vertex)
        distance_value = int(dist[vertex]) if float(dist[vertex]).is_integer() else dist[vertex]
        lines.append(f"{vertex}, {distance_value}, {path}")
    return "\n".join(lines)

def output_path_for_input_MattKeith(input_path: str) -> str:
    # build the output filename inside the outputs/ folder
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.basename(input_path)
    return os.path.join("outputs", f"output_{base_name}")

def solve_file_MattKeith(input_path: str, source: int = 0) -> str:
    # run Bellman‑Ford on one input file and write the output
    n, matrix = read_input_file_MattKeith(input_path)
    edges = matrix_to_edges_MattKeith(matrix)
    has_negative_cycle, dist, pred = bellman_ford_final_MattKeith(n, edges, source)
    output_text = format_output_MattKeith(n, has_negative_cycle, dist, pred, source)

    out_path = output_path_for_input_MattKeith(input_path)
    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
        outfile.write("\n")

    return output_text

def main(argv: Optional[Sequence[str]] = None) -> int:
    # usage: python BellmanFordDP.py file1 file2 ...
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python BellmanFordDP.py <inputfile1> [<inputfile2> ...]", file=sys.stderr)
        return 1

    for input_path in argv:
        try:
            solve_file_MattKeith(input_path, source=0)
        except Exception as exc:  # pragma: no cover - defensive for marking script use
            print(f"Error processing {input_path}: {exc}", file=sys.stderr)
            return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())