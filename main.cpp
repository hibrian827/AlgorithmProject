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

int tsp_solve(const Matrix& w)
{
  // Create and use implemented solvers
  // This is an example tsp_solve function.
  // you can modify this tsp_solve function,
  // but make sure to call print_result()
  // Make sure 'tour' be 0-indexed, and include the starting node at the end.
  auto solver = make_unique<TwoStepGreedySolver>(w);
  auto [best_cost, tour] = solver->solve(0);
  print_result(best_cost, tour);
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
