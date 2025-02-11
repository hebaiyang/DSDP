#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

// 订单板结构体
struct OrderPlate {
    int h, w, l;
    int arrival_time, deadline;
};

// 板坯结构体
struct Slab {
    int H, W, L;
};

// 轧制方法结构体
struct RollingMethod {
    double C1, C2, C3, C4;
};

// 读取并解析文件数据
bool read_test_case(const std::string& filename,
                    std::vector<std::vector<OrderPlate>>& order_plates,
                    std::vector<Slab>& slabs,
                    std::vector<RollingMethod>& rolling_methods) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return false;
    }

    // 读取批次数量
    int n_batches;
    file >> n_batches;

    // 读取每个批次的订单板数据
    order_plates.resize(n_batches);
    for (int i = 0; i < n_batches; ++i) {
        int q; // 当前批次的订单板数量
        file >> q;
        order_plates[i].resize(q);

        // 读取订单板的厚度、宽度、长度、到达时间、截止时间
        for (int j = 0; j < q; ++j) {
            OrderPlate plate;
            file >> plate.h >> plate.w >> plate.l >> plate.arrival_time >> plate.deadline;
            order_plates[i][j] = plate;
        }
    }

    // 读取板坯的数量并解析数据
    int slab_set_size;
    file >> slab_set_size;
    slabs.resize(slab_set_size);

    for (int i = 0; i < slab_set_size; ++i) {
        Slab slab;
        file >> slab.H >> slab.W >> slab.L;
        slabs[i] = slab;
    }

    // 读取轧制方法的数量并解析数据
    int rolling_method_set_size;
    file >> rolling_method_set_size;
    rolling_methods.resize(rolling_method_set_size);

    for (int i = 0; i < rolling_method_set_size; ++i) {
        RollingMethod method;
        file >> method.C1 >> method.C2 >> method.C3 >> method.C4;
        rolling_methods[i] = method;
    }

    file.close();
    return true;
}

int main() {
    // 文件路径
    std::string filename = "test_case.txt";

    // 存储读取的数据
    std::vector<std::vector<OrderPlate>> order_plates;
    std::vector<Slab> slabs;
    std::vector<RollingMethod> rolling_methods;

    // 读取测试数据并将结果保存在引用参数中
    if (read_test_case(filename, order_plates, slabs, rolling_methods)) {
        std::cout << "Data successfully read from the file." << std::endl;

        // 输出读取的数据（仅作为示例）
        std::cout << "Order Plates Data:" << std::endl;
        for (size_t i = 0; i < order_plates.size(); ++i) {
            std::cout << "Batch " << i + 1 << ":" << std::endl;
            for (size_t j = 0; j < order_plates[i].size(); ++j) {
                std::cout << "  Plate " << j + 1 << ": "
                          << order_plates[i][j].h << " "
                          << order_plates[i][j].w << " "
                          << order_plates[i][j].l << " "
                          << order_plates[i][j].arrival_time << " "
                          << order_plates[i][j].deadline << std::endl;
            }
        }

        std::cout << "\nSlabs Data:" << std::endl;
        for (size_t i = 0; i < slabs.size(); ++i) {
            std::cout << "  Slab " << i + 1 << ": "
                      << slabs[i].H << " "
                      << slabs[i].W << " "
                      << slabs[i].L << std::endl;
        }

        std::cout << "\nRolling Methods Data:" << std::endl;
        for (size_t i = 0; i < rolling_methods.size(); ++i) {
            std::cout << "  Method " << i + 1 << ": "
                      << rolling_methods[i].C1 << " "
                      << rolling_methods[i].C2 << " "
                      << rolling_methods[i].C3 << " "
                      << rolling_methods[i].C4 << std::endl;
        }
    } else {
        std::cerr << "Failed to read data from the file." << std::endl;
    }

    return 0;
}
