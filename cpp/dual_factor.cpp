#include <iostream>
#include <vector>
#include <ilcplex/ilocplex.h>  // 包含 CPLEX 库

ILOSTLBEGIN

struct OrderPlate {
    int h, w, l;          // 厚度、宽度、长度
    int arrival_time, deadline;
    int batch_id;         // 批次ID，表示属于哪个批次
    int index;
};

std::vector<double> solveLinear(std::vector<std::vector<std::vector<std::vector<std::vector<OrderPlate>>>>> scheme_pool,
    std::vector<std::vector<std::vector<std::vector<double>>>> benefit_pool, int order_plates_numb, int T, int J, int G, int Q) {
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> coeff;
    for(int i=0;i<scheme_pool.size();i++) {
        coeff.push_back(std::vector<std::vector<std::vector<std::vector<int>>>>());
        for(int j=0;j<scheme_pool[i].size();j++) {
            coeff[i].push_back(std::vector<std::vector<std::vector<int>>>());
            for(int k=0;k<scheme_pool[i][j].size();k++) {
                coeff[i][j].push_back(std::vector<std::vector<int>>());
                for(int g=0;g<scheme_pool[i][j][k].size();g++) {
                    coeff[i][j][k].push_back(std::vector<int>(order_plates_numb));
                    for(int e=0;e<scheme_pool[i][j][k][g].size();e++) {
                        coeff[i][j][k][g][scheme_pool[i][j][k][g][e].index]=1;
                    }
                }
            }
        }
    }


    IloEnv env;
    IloModel model(env);
    IloCplex cplex(model);

    // 创建一个决策变量数组
    IloArray<IloArray<IloArray<IloArray<IloNumVar>>>> x(env, T);
    for (int t = 0; t <= T; ++t) {
        x[t] = IloArray<IloArray<IloArray<IloNumVar>>>(env, J);
        for (int j = 0; j < J; ++j) {
            x[t][j] = IloArray<IloArray<IloNumVar>>(env, G);
            for (int g = 0; g < G; ++g) {
                x[t][j][g] = IloArray<IloNumVar>(env, Q);
                for (int q = 0; q < Q; ++q) {
                    std::string varName = "x_" + std::to_string(t) + "_" + std::to_string(j) + "_" + std::to_string(g) + "_" + std::to_string(q);
                    x[t][j][g][q] = IloNumVar(env, 0, 1, varName.c_str());
                    model.addVariable(x[t][j][g][q]);
                }
            }
        }
    }


    // 目标函数：最大化选择的设计方案的总利润
    IloExpr obj(env);
    for (int t = 0; t <= T; ++t) {
        for (int j = 0; j < J; ++j) {
            for (int g = 0; g < G; ++g) {
                for (int q = 0; q < Q; ++q) {
                    obj += benefit_pool[t][j][g][q] * x[t][j][g][q];  // 利润 * 决策变量
                }
            }
        }
    }
    model.addMaximize(obj);
    obj.end();

    for(int e=0;e<order_plates_numb;e++) {
        IloExpr constraint1(env);
        for(int i=0;i<coeff.size();i++) {
            for(int j=0;j<coeff[i].size();j++) {
                for(int k=0;k<coeff[i][j].size();k++) {
                    for(int g=0;g<coeff[i][j][k].size();g++) {
                        constraint1 += coeff[i][j][k][g] * x[i][j][k][g];
                    }
                }
            }
            model.addConstraint(constraint1 <= 1);
            constraint1.end();
        }
    }


    IloExpr constraint2(env);
    for (int t = 0; t <= T; ++t) {
        for (int j = 0; j < J; ++j) {
            for (int g = 0; g < G; ++g) {
                for (int q = 0; q < Q; ++q) {
                    constraint2 += x[t][j][g][q];
                }
            }
        }
    }
    model.addConstraint(constraint2 <= T+1);
    constraint2.end();

    for (int t = 0; t <= T; ++t) {
        for (int j = 0; j < J; ++j) {
            IloExpr constraint3(env);
            for (int g = 0; g < G; ++g) {
                for (int q = 0; q < Q; ++q) {
                    constraint3 += x[t][j][g][q];
                }
            }
            model.addConstraint(constraint3 <= 1);
            constraint3.end();
        }
    }

    for (int t = 0; t <= T; t++) {
        for (int g = 0; g < G; g++) {
            IloExpr constraint4(env);
            for (int j = 0; j < J; j++) {
                for (int q = 0; q < Q; ++q) {
                    constraint4 += x[t][j][g][q];
                }
            }
            model.addConstraint(constraint4 <= 1);
            constraint4.end();
        }
    }

    // 求解模型
    if (!cplex.solve()) {
        std::cerr << "Failed to solve the LP model!" << std::endl;
        return {};
    }

    // 获取对偶变量的值
    std::vector<double> dualValues;
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < J; ++j) {
            for (int g = 0; g < G; ++g) {
                for (int q = 0; q < Q; ++q) {
                    double dualValue = cplex.getDuals(x[t][j][g][q]);
                    dualValues.push_back(dualValue);
                }
            }
        }
    }

    cplex.end();
    env.end();
    return dualValues;
}

extern "C" {std::vector<double> solveLinear(
    std::vector<std::vector<std::vector<std::vector<std::vector<OrderPlate>>>>> orderPlateArray,
    std::vector<std::vector<std::vector<std::vector<double>>>> benefitArray, int order_plates_num, int T, int J, int G, int Q
);}
