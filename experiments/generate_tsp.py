"""Generate a TSP split with exact optimal tours (brute force, n <= 9).

    python experiments/generate_tsp.py --n-cities 5 --samples 1000 --out data/tsp5.pkl

The output matches the format the run scripts expect: a pickle with
``coordinates`` [N, n, 2], ``optimal_tours`` [N, n], and ``optimal_lengths`` [N].
Larger instances need an external solver such as LKH (e.g. the ``elkai`` package).
"""
import argparse
import pickle

import numpy as np

from mp_qgnns.data_io.tsp import optimal_tour, tour_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cities", type=int, default=5)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    coords = rng.random((args.samples, args.n_cities, 2))
    tours = np.zeros((args.samples, args.n_cities), dtype=np.int64)
    lengths = np.zeros(args.samples)
    for i, c in enumerate(coords):
        tour, _ = optimal_tour(c)
        tours[i] = tour
        lengths[i] = tour_length(c, tour)
    with open(args.out, "wb") as f:
        pickle.dump({"coordinates": coords, "optimal_tours": tours,
                     "optimal_lengths": lengths}, f)
    print(f"wrote {args.out}: {args.samples} instances of TSP-{args.n_cities}")


if __name__ == "__main__":
    main()
