#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <string>

struct OrderPlate {
    int h, w, l;          // 厚度、宽度、长度
    int arrival_time, deadline;
    int batch_id;         // 批次ID，表示属于哪个批次
};

struct Slab {
    int H, W, L;
};

struct RollingMethod {
    double C1, C2, C3, C4;
};

// 读取并解析文件数据
int read_test_case(const std::string& filename,
                    std::vector<std::vector<OrderPlate>>& order_plates,
                    std::vector<Slab>& slabs,
                    std::vector<RollingMethod>& rolling_methods) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 0;
    }

    std::string line;
    int n_batches;
    file >> n_batches;  // 读取批次数量

    order_plates.resize(n_batches);  // 根据批次数量初始化 order_plates 数组

    // 读取每个批次的订单板数据
    for (int k = 0; k < n_batches; ++k) {
        int q;
        file >> q;  // 读取当前批次的订单板数量

        for (int i = 0; i < q; ++i) {
            OrderPlate plate;
            file >> plate.h >> plate.w >> plate.l >> plate.arrival_time >> plate.deadline;
            order_plates[k].push_back(plate);  // 将订单板添加到对应批次
        }
    }

    // 读取板坯数据
    int slab_set_size;
    file >> slab_set_size;
    slabs.resize(slab_set_size);  // 初始化 slab 数组

    for (int j = 0; j < slab_set_size; ++j) {
        Slab slab;
        file >> slab.H >> slab.W >> slab.L;
        slabs.push_back(slab);  // 将板坯数据存入 slabs 数组
    }

    // 读取轧制方法数据
    int rolling_method_set_size;
    file >> rolling_method_set_size;
    rolling_methods.resize(rolling_method_set_size);  // 初始化 rolling_methods 数组

    for (int g = 0; g < rolling_method_set_size; ++g) {
        RollingMethod method;
        file >> method.C1 >> method.C2 >> method.C3 >> method.C4;
        rolling_methods.push_back(method);  // 将轧制方法存入 rolling_methods 数组
    }

    file.close();

    return 1;
}

int read_data(const std::string& filename, std::vector<std::vector<OrderPlate>>& order_plates,
    std::vector<Slab>& slabs, std::vector<RollingMethod>& rolling_methods) {
    // std::string filename = "test_case.txt";
    //
    // // 创建存储数据的数组
    // std::vector<std::vector<OrderPlate>> order_plates;
    // std::vector<Slab> slabs;
    // std::vector<RollingMethod> rolling_methods;

    // 读取文件并将数据保存到数组
    if(read_test_case(filename, order_plates, slabs, rolling_methods)) {
        return 1;
    }

    return 0;
}


// 生成随机数
int generate_random(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

double generate_random_double(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// 生成测试数据并保存到文件
void generate_test_case(const std::string& filename, int n_batches) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // 用于存储每个批次的订单板
    std::vector<std::vector<OrderPlate>> batches;

    // 生成订单板数据
    for (int k = 0; k < n_batches; ++k) {
        int q = generate_random(30, 300); // 每个批次的订单板数量
        int AT_k = generate_random(0, 2505500); // 到达时间
        int deadline_k = generate_random(AT_k + 86400, 2592000); // 截止时间

        // 创建当前批次的订单板
        std::vector<OrderPlate> batch;

        for (int i = 0; i < q; ++i) {
            int h = 8;//generate_random(5, 10);        // 厚度
            int w = generate_random(500, 600);     // 宽度
            int l = generate_random(w + 50, 700);  // 长度

            OrderPlate plate = {h, w, l, AT_k, deadline_k, k};
            batch.push_back(plate);
        }

        batches.push_back(batch);
    }

    // 按照到达时间排序批次，使用lambda表达式
    std::sort(batches.begin(), batches.end(), [](const std::vector<OrderPlate>& batch1, const std::vector<OrderPlate>& batch2) {
        return batch1.front().arrival_time < batch2.front().arrival_time;
    });

    // 写入文件
    file << n_batches << std::endl;
    for (const auto& batch : batches) {
        file << batch.size() << std::endl;

        // 写入当前批次的所有订单板数据
        for (const auto& plate : batch) {
            file << plate.h << " " << plate.w << " " << plate.l << " "
                 << plate.arrival_time << " " << plate.deadline << std::endl;
        }

        file << std::endl; // 每个批次之间空一行
    }

    // 生成板坯数据
    int slab_set_size = generate_random(5, 30);
    file << slab_set_size << std::endl;
    for (int j = 0; j < slab_set_size; ++j) {
        int H = generate_random(15, 20);
        int W = generate_random(700, 800);
        int L = generate_random(W + 100, 1000);
        file << H << " " << W << " " << L << std::endl;
    }

    // 生成轧制方法数据
    int rolling_method_set_size = generate_random(2, 10);
    file << rolling_method_set_size << std::endl;
    for (int g = 0; g < rolling_method_set_size; ++g) {
        double C1 = generate_random_double(10, 50);
        double C2 = generate_random_double(20, 80);
        double C3 = generate_random_double(0.05, 0.08);
        double C4 = generate_random_double(50, 200);
        file << C1 << " " << C2 << " " << C3 << " " << C4 << std::endl;
    }

    file.close();
}


void generate_data(std::string filename, int n_batches) {
    // // 设置生成的批次数量
    // int n_batches = 10;
    //
    // // 设置保存文件路径
    // std::string filename = "test_case.txt";

    // 生成测试实例
    generate_test_case(filename, n_batches);

    std::cout << "Test case has been generated and saved to " << filename << "\n";
}

int main() {
    std::string filename = "test_case.txt";
    int n_batches = 13;  // 总批次数量

    // 生成测试实例并保存到文件
    generate_data(filename, n_batches);

    std::vector<std::vector<OrderPlate>> order_plates;
    std::vector<Slab> slabs;
    std::vector<RollingMethod> rolling_methods;
    read_data(filename, order_plates, slabs, rolling_methods);

    std::cout << "Test case generated and saved to " << filename << std::endl;
    return 0;
}


extern "C" {
    int read_data(std::string filename, std::vector<std::vector<OrderPlate>>& order_plates,std::vector<Slab>& slabs, std::vector<RollingMethod>& rolling_methods);
}
