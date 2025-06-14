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
#include <cfloat>
#include <random>
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
  explicit SimulatedAnnealingSolver(const Matrix& w) : TSPSolver(w), rng(random_device{}()) {}

  pair<long long, vector<int>> solve(int start = 0) override {
    vector<int> current = initial_solution(start);
    vector<int> best = current;

    long long current_cost = tour_cost(current);
    long long best_cost = current_cost;

    double T = 1e20;
    double alpha = 0.99999;
    int max_iter = 11000000;
    
    int iter = 0;
    uniform_real_distribution<double> prob_dist(0.0, 1.0);

    while (T > 1e-100 && iter < max_iter) {
      vector<int> neighbor = generate_neighbor(current);
      long long neighbor_cost = tour_cost(neighbor);
      long long delta = neighbor_cost - current_cost;

      if (delta < 0 || prob_dist(rng) < exp(-delta / T)) {
        current = neighbor;
        current_cost = neighbor_cost;

        if (current_cost < best_cost) {
          best = current;
          best_cost = current_cost;
        }
      }

      if(best_cost < INF) iter++;
      T *= alpha;
    }
    best.push_back(start);

    return {best_cost, best};
  }

private:
  mt19937 rng;

  vector<int> initial_solution(int start) {
    vector<bool> visited(n, false);
    vector<int> tour;
    tour.push_back(start);
    visited[start] = true;

    for (int i = 1; i < n; ++i) {
      int last = tour.back();
      int next = -1;
      int best_dist = INF + 1;
      for (int j = 0; j < n; ++j) {
        if (!visited[j] && W[last][j] < best_dist) {
          best_dist = W[last][j];
          next = j;
        }
      }
      tour.push_back(next);
      visited[next] = true;
    }
    return tour;
  }

  long long tour_cost(const vector<int>& path) const {
    long long cost = 0;
    for (int i = 0; i < n; ++i) {
      int from = path[i];
      int to = path[(i + 1) % n];
      cost += W[from][to];
    }
    return cost;
  }

  vector<int> generate_neighbor(const vector<int>& path) {
    vector<int> neighbor = path;
    uniform_int_distribution<int> dist(1, n - 1);
    int i = dist(rng), j = dist(rng);
    while (i == j) j = dist(rng);
    if (i > j) swap(i, j);
    reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
    return neighbor;
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
  if (n <= 10) {
    solver = new HeldKarpSolver(w);
  } else solver = new SimulatedAnnealingSolver(w);

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