# Contexto Benchmark Results

Results show the average number of attempts needed to solve each game (lower is better).

| Strategy                  | Game 1   | Game 2    | Game 3    | Game 4   | Game 5   |   Game Avg |
|:--------------------------|:---------|:----------|:----------|:---------|:---------|-----------:|
| context_search            | 81 ± 18  | 161 ± 166 | 58 ± 25   | 30 ± 8   | 84 ± 11  |         83 |
| nearest_descent           | 120 ± 65 | 376 ± 26  | 129 ± 88  | 223 ± 6  | 58 ± 6   |        181 |
| context_search_stochastic | 162 ± 66 | 278 ± 175 | 258 ± 30  | 197 ± 82 | 132 ± 40 |        205 |
| nearest_descent_10_starts | 212 ± 25 | 374 ± 11  | 135 ± 114 | 157 ± 91 | 225 ± 2  |        221 |