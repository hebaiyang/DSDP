//
// Created by dell on 25-2-11.
//
#include <iostream>
#include <vector>
#include <random>

struct Plate {
    double thickness;
    double width;
    double length;
};

struct Order {
    double thickness;
    double width;
    double length;
    double profit;
};

struct RollingMethod {
    double c1, c2, c3, c4;
};

std::vector<Plate> generatePlates(int num_plates) {
    std::vector<Plate> plates;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 10.0);  // Random plate dimensions

    for (int i = 0; i < num_plates; ++i) {
        Plate p = {dis(gen), dis(gen), dis(gen)};
        plates.push_back(p);
    }
    return plates;
}

std::vector<Order> generateOrders(int num_orders) {
    std::vector<Order> orders;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 5.0);  // Random order dimensions

    for (int i = 0; i < num_orders; ++i) {
        Order o = {dis(gen), dis(gen), dis(gen), dis(gen)};
        orders.push_back(o);
    }
    return orders;
}

std::vector<RollingMethod> generateRollingMethods(int num_methods) {
    std::vector<RollingMethod> methods;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.01, 0.5);  // Random rolling method parameters

    for (int i = 0; i < num_methods; ++i) {
        RollingMethod rm = {dis(gen), dis(gen), dis(gen), dis(gen)};
        methods.push_back(rm);
    }
    return methods;
}

extern "C" {
    std::vector<Plate> *generate_plates(int num_plates) {
        return new std::vector<Plate>(generatePlates(num_plates));
    }

    std::vector<Order> *generate_orders(int num_orders) {
        return new std::vector<Order>(generateOrders(num_orders));
    }

    std::vector<RollingMethod> *generate_rolling_methods(int num_methods) {
        return new std::vector<RollingMethod>(generateRollingMethods(num_methods));
    }
}
