#include <iostream>
#include <vector>

struct Order {
    double volume;
    double profit;
};

double knapsack(const std::vector<Order>& orders, double capacity) {
    int n = orders.size();
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(static_cast<int>(capacity) + 1, 0));

    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= capacity; ++w) {
            if (orders[i - 1].volume <= w) {
                dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - static_cast<int>(orders[i - 1].volume)] + orders[i - 1].profit);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][static_cast<int>(capacity)];
}

extern "C" {
    double knapsack_cpp(std::vector<Order>& orders, double capacity);
}
