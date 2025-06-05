//============================================================================
// SNU Spring 2025 Introduction to Algorithm Final Project
//============================================================================
// Travelling Salesman Problem – Skeleton Implementation (C++17)
//  ---------------------------------------------------------------------------
//   * ONLY edit the TODO SECTION. Modifying other sections may cause error on submission
//   * Locked sections handle all I/O and must not be modified.
// 
// How to run locally:
//   g++ -std=c++17 -O2 main.cpp -o tsp
//   ./tsp <graph_file>
//============================================================================

#include <bits/stdc++.h>
using namespace std;

/*=====================  LOCKED SECTION  (do not edit)  =====================*/
constexpr int INF = 1e9;
using Matrix = vector<vector<int>>;
int tsp_solve(const Matrix&);

static Matrix read_graph(istream& in) {
  int n, m; if (!(in >> n >> m)) return {};
  Matrix w(n, vector<int>(n, INF));
  for (int i = 0; i < n; ++i) w[i][i] = 0;
  for (int i = 0; i < m; ++i) {
    int u, v, c; in >> u >> v >> c; --u; --v;
    w[u][v] = w[v][u] = c;
  }
  return w;
}

static void print_result(long long cost, const vector<int>& t)
{
  if (cost >= INF) { cout << -1 << '\n'; return; }
  cout << cost;
  for (int v : t) cout << ' ' << v + 1;
  cout << '\n';
}

// ---------------------------------------------------------------------------
// Abstract base class for any TSP solver.
// You can use this TSPSolver class, but it's okay if you don't use this.
// ---------------------------------------------------------------------------
class TSPSolver {
public:
  explicit TSPSolver(const Matrix& w) : W(w), n(static_cast<int>(w.size())) {}
  virtual ~TSPSolver() = default;

  // Returns {best_cost, tour_vertices}
  virtual pair<long long, vector<int>> solve(int start = 0) = 0;

protected:
  const Matrix& W;
  int n;
};

// Two-step greedy solver for TSP
class TwoStepGreedySolver : public TSPSolver {
public:
  using TSPSolver::TSPSolver;
  pair<long long, vector<int>> solve(int start = 0) override;
};

// Main function
int main(int argc, char* argv[]) {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  if (argc > 1) {                          // file list mode
    for (int i = 1; i < argc; ++i) {
      ifstream fin(argv[i]);
      if (!fin) { cerr << "Cannot open " << argv[i] << '\n'; continue; }
      Matrix W = read_graph(fin);
      tsp_solve(W);
    }
  } else {                                 // stdin single graph
    Matrix W = read_graph(cin);
    tsp_solve(W);
  }
}
/*===================  END LOCKED SECTION  ==================================*/



/*=====================  TODO SECTION  (students edit)  =====================*/
// Goal
// 1. Complete Two-Step greedy solver
// 2. Improve TwoStepGreedySolver or implement new TSP algorithm for better optimality and efficiency

class HeldKarpSolver : public TSPSolver {
public:
  using TSPSolver::TSPSolver;

  pair<long long, vector<int>> solve(int start = 0) override {
    const long long INF = numeric_limits<long long>::max() / 2;
    int FULL = (1 << n);
    vector<vector<long long>> dp(FULL, vector<long long>(n, INF));
    vector<vector<int>> parent(FULL, vector<int>(n, -1));

    // Base case
    dp[1 << start][start] = 0;

    // Iterate over subsets of increasing size
    for (int mask = 0; mask < FULL; ++mask) {
      for (int u = 0; u < n; ++u) {
        if (!(mask & (1 << u))) continue; // u not in mask
        for (int v = 0; v < n; ++v) {
          if ((mask & (1 << v)) || W[v][u] >= INF) continue; // v already in mask or no edge
          int prev_mask = mask | (1 << v);
          long long new_cost = dp[mask][u] + W[u][v];
          if (new_cost < dp[prev_mask][v]) {
            dp[prev_mask][v] = new_cost;
            parent[prev_mask][v] = u;
          }
        }
      }
    }

    // Close the tour
    long long best_cost = INF;
    int last_city = -1;
    for (int u = 0; u < n; ++u) {
      if (u == start || W[u][start] >= INF) continue;
      long long cost = dp[FULL - 1][u] + W[u][start];
      if (cost < best_cost) {
        best_cost = cost;
        last_city = u;
      }
    }

    // Reconstruct path
    vector<int> path;
    int mask = FULL - 1;
    int curr = last_city;
    while (curr != -1) {
      path.push_back(curr);
      int temp = parent[mask][curr];
      mask ^= (1 << curr);
      curr = temp;
    }
    reverse(path.begin(), path.end());
    path.push_back(start); // to complete the tour

    return {best_cost, path};
  }
};

class SimulatedAnnealingSolver : public TSPSolver {
public:
  using TSPSolver::TSPSolver;

  pair<long long, vector<int>> solve(int start = 0) override {
    const long long INF = numeric_limits<long long>::max() / 2;

    // Random engine
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> real_dist(0.0, 1.0);

    // Initial tour (0 → 1 → 2 → ... → n-1 → 0)
    vector<int> current_tour(n);
    iota(current_tour.begin(), current_tour.end(), 0);

    auto tour_cost = [&](const vector<int>& tour) -> long long {
      long long cost = 0;
      for (int i = 0; i < n; ++i) {
        int from = tour[i];
        int to = tour[(i + 1) % n];
        if (W[from][to] >= INF) return INF;
        cost += W[from][to];
      }
      return cost;
    };

    auto swap_two = [&](vector<int>& tour) {
      int i = rng() % n;
      int j = rng() % n;
      while (j == i) j = rng() % n;
      swap(tour[i], tour[j]);
    };

    // Parameters
    double T = 10000.0;
    double cooling_rate = 0.995;
    int iterations_per_temp = 100;

    vector<int> best_tour = current_tour;
    long long best_cost = tour_cost(current_tour);

    while (T > 1e-4) {
      for (int it = 0; it < iterations_per_temp; ++it) {
        vector<int> new_tour = current_tour;
        swap_two(new_tour);
        long long new_cost = tour_cost(new_tour);
        long long curr_cost = tour_cost(current_tour);

        double acceptance_prob = exp((curr_cost - new_cost) / T);

        if (new_cost < curr_cost || real_dist(rng) < acceptance_prob) {
          current_tour = new_tour;
          if (new_cost < best_cost) {
            best_cost = new_cost;
            best_tour = new_tour;
          }
        }
      }
      T *= cooling_rate;
    }

    // Ensure the tour starts at `start`
    auto it = find(best_tour.begin(), best_tour.end(), start);
    rotate(best_tour.begin(), it, best_tour.end());
    best_tour.push_back(start); // complete the cycle

    return {best_cost, best_tour};
  }
};

class OneTreeSparsifierSolver : public TSPSolver {
public:
  using TSPSolver::TSPSolver;

  vector<pair<int, int>> primMST(const Matrix& W, int skip) {
    const long long INF = numeric_limits<long long>::max() / 2;
    int n = W.size();
    vector<bool> inMST(n, false);
    vector<long long> key(n, INF);
    vector<int> parent(n, -1);
    vector<pair<int, int>> mst_edges;

    key[(skip + 1) % n] = 0;

    for (int count = 0; count < n - 1; ++count) {
      int u = -1;
      long long minKey = INF;
      for (int v = 0; v < n; ++v) {
        if (!inMST[v] && key[v] < minKey && v != skip) {
          u = v;
          minKey = key[v];
        }
      }

      if (u == -1) break;
      inMST[u] = true;
      if (parent[u] != -1)
        mst_edges.emplace_back(parent[u], u);

      for (int v = 0; v < n; ++v) {
        if (!inMST[v] && v != skip && W[u][v] < key[v]) {
          key[v] = W[u][v];
          parent[v] = u;
        }
      }
    }
    return mst_edges;
  }

  pair<long long, vector<int>> solve(int start = 0) override {
    Matrix sparseW(n, vector<int>(n, INF));

    // 1-tree construction: start is the special node
    // 1. Find MST excluding `start`
    auto mst_edges = primMST(W, start);
    for (auto [u, v] : mst_edges) sparseW[u][v] = sparseW[v][u] = W[u][v];

    // 2. Add two smallest edges connecting `start`
    vector<pair<long long, int>> start_edges;
    for (int i = 0; i < n; ++i) {
      if (i == start) continue;
      if (W[start][i] < INF) start_edges.emplace_back(W[start][i], i);
    }
    sort(start_edges.begin(), start_edges.end());
    for (int i = 0; i < min(2, (int)start_edges.size()); ++i) {
      int v = start_edges[i].second;
      sparseW[start][v] = sparseW[v][start] = W[start][v];
    }

    // Now pass the sparse matrix to a downstream solver
    SparseDownstreamSolver sparseSolver(sparseW);
    return sparseSolver.solve(start);
  }

private:
  // You can replace this with any real solver
  class SparseDownstreamSolver : public TSPSolver {
  public:
    using TSPSolver::TSPSolver;
    pair<long long, vector<int>> solve(int start = 0) override {
      // Placeholder: return trivial tour
      vector<int> tour(n);
      iota(tour.begin(), tour.end(), 0);
      rotate(tour.begin(), find(tour.begin(), tour.end(), start), tour.end());
      tour.push_back(start);
      long long cost = 0;
      for (int i = 0; i < n; ++i)
        cost += W[tour[i]][tour[(i + 1) % n]];
      return {cost, tour};
    }
  };
};

class ChristofidesSolver : public TSPSolver {
public:
  using TSPSolver::TSPSolver;

  pair<long long, vector<int>> solve(int start = 0) override {
    // Step 1: Compute MST using Prim's algorithm
    vector<pair<int, int>> mst = primMST();

    // Step 2: Find vertices with odd degree
    vector<int> degree(n, 0);
    for (auto [u, v] : mst) {
      degree[u]++;
      degree[v]++;
    }

    vector<int> odd_vertices;
    for (int i = 0; i < n; ++i) {
      if (degree[i] % 2 == 1)
        odd_vertices.push_back(i);
    }

    // Step 3: Find minimum weight perfect matching among odd-degree vertices
    vector<pair<int, int>> matching = greedyPerfectMatching(odd_vertices);

    // Step 4: Create multigraph (MST + matching)
    vector<multiset<int>> multigraph(n);
    for (auto [u, v] : mst) {
      multigraph[u].insert(v);
      multigraph[v].insert(u);
    }
    for (auto [u, v] : matching) {
      multigraph[u].insert(v);
      multigraph[v].insert(u);
    }

    // Step 5: Find Eulerian tour
    vector<int> eulerian_tour;
    findEulerianTour(start, multigraph, eulerian_tour);

    // Step 6: Shortcut to Hamiltonian cycle
    vector<bool> visited(n, false);
    vector<int> hamiltonian_path;
    for (int v : eulerian_tour) {
      if (!visited[v]) {
        visited[v] = true;
        hamiltonian_path.push_back(v);
      }
    }
    hamiltonian_path.push_back(hamiltonian_path[0]); // complete the cycle

    long long total_cost = 0;
    for (size_t i = 0; i + 1 < hamiltonian_path.size(); ++i)
      total_cost += W[hamiltonian_path[i]][hamiltonian_path[i + 1]];

    return {total_cost, hamiltonian_path};
  }

private:
  vector<pair<int, int>> primMST() {
    const long long INF = numeric_limits<long long>::max() / 2;
    vector<long long> key(n, INF);
    vector<int> parent(n, -1);
    vector<bool> inMST(n, false);
    key[0] = 0;

    for (int count = 0; count < n - 1; ++count) {
      int u = -1;
      long long minKey = INF;
      for (int v = 0; v < n; ++v) {
        if (!inMST[v] && key[v] < minKey) {
          u = v;
          minKey = key[v];
        }
      }

      if (u == -1) break;
      inMST[u] = true;

      for (int v = 0; v < n; ++v) {
        if (!inMST[v] && W[u][v] < key[v]) {
          key[v] = W[u][v];
          parent[v] = u;
        }
      }
    }

    vector<pair<int, int>> mst;
    for (int v = 1; v < n; ++v)
      if (parent[v] != -1)
        mst.emplace_back(parent[v], v);
    return mst;
  }

  // Greedy matching — not guaranteed MWPM, but fast
  vector<pair<int, int>> greedyPerfectMatching(vector<int> nodes) {
    const long long INF = numeric_limits<long long>::max() / 2;
    vector<pair<int, int>> matching;
    vector<bool> used(n, false);
    sort(nodes.begin(), nodes.end());

    while (!nodes.empty()) {
      int u = nodes.back();
      nodes.pop_back();
      if (used[u]) continue;

      long long best_w = INF;
      int best_v = -1;
      for (int v : nodes) {
        if (!used[v] && W[u][v] < best_w) {
          best_w = W[u][v];
          best_v = v;
        }
      }
      if (best_v != -1) {
        matching.emplace_back(u, best_v);
        used[u] = used[best_v] = true;
        nodes.erase(find(nodes.begin(), nodes.end(), best_v));
      }
    }

    return matching;
  }

  void findEulerianTour(int u, vector<multiset<int>>& graph, vector<int>& path) {
    while (!graph[u].empty()) {
      int v = *graph[u].begin();
      graph[u].erase(graph[u].begin());
      graph[v].erase(graph[v].find(u));
      findEulerianTour(v, graph, path);
    }
    path.push_back(u);
  }
};

int tsp_solve(const Matrix& w)
{
  // Create and use implemented solvers
  // This is an example tsp_solve function.
  // you can modify this tsp_solve function,
  // but make sure to call print_result()
  // Make sure 'tour' be 0-indexed, and include the starting node at the end.
  
  /* Greedy */
  // auto solver = make_unique<TwoStepGreedySolver>(w);

  /* Improvement */
  int n = w.size();
  int non_inf_edges = 0;
  int total_possible_edges = n * (n - 1);

  // Count the non-infinite, non-diagonal edges
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j && w[i][j] != INF) {
        non_inf_edges++;
      }
    }
  }

  double density = static_cast<double>(non_inf_edges) / total_possible_edges;

  // Decision logic
  TSPSolver* solver = nullptr;
  if (n <= 5) {
    solver = new HeldKarpSolver(w);
  } else if (n <= 30) {
    if (density > 0.8) {
      solver = new SimulatedAnnealingSolver(w);
    } else {
      // solver = new OneTreeSparsifierSolver(w);
      solver = new SimulatedAnnealingSolver(w);
    }
  } else {
    if (density > 0.8) {
      solver = new ChristofidesSolver(w);
    } else {
      solver = new ChristofidesSolver(w);
      // TODO
      // Sparse and large: using Ant Colony with pruning or hierarchical strategy
    }
  }

  if(solver != nullptr) {
    auto [best_cost, tour] = solver->solve(0);
    print_result(best_cost, tour);
  }
  else print_result(-1, {});
  return 0;
}

// Two-step greedy algorithm with lookahead implementation
pair<long long, vector<int>> TwoStepGreedySolver::solve(int start) {
  vector<int> tour;
  vector<bool> visited(n, false);
  long long total_cost = 0;
  int current = start;
  tour.push_back(current);
  visited[current] = true;

  for (int step = 1; step < n; ++step) {
    int next_node = -1;
    long long best_total = INF;

    for (int a = 0; a < n; ++a) {
      if (visited[a] || W[current][a] == INF)
        continue;

      // Find minimum distance from A to any other unvisited B
      long long min_second_step = INF;
      bool has_unvisited = false;

      for (int b = 0; b < n; ++b) {
        if (b != a && !visited[b] && W[a][b] != INF) {
          if (!has_unvisited || W[a][b] < min_second_step) {
            min_second_step = W[a][b];
            has_unvisited = true;
          }
        }
      }

      if (!has_unvisited)
        min_second_step = 0;

      long long total = W[current][a] + min_second_step;

      if (total < best_total || (total == best_total && (next_node == -1 || a < next_node))) {
        best_total = total;
        next_node = a;
      }
    }

    if (next_node == -1)
      return {INF, {}};

    total_cost += W[current][next_node];
    current = next_node;
    visited[current] = true;
    tour.push_back(current);
  }

  if (W[current][start] == INF)
    return {INF, {}};

  total_cost += W[current][start];
  tour.push_back(start);

  return {total_cost, tour};
}