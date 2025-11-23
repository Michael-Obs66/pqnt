// main.cpp
#include "pqnt.hpp"
#include <iostream>
#include <vector>
#include <iomanip>   // setprecision
#include <cstdlib>   // atof, atoi

using namespace std;

int main(int argc, char** argv) {
    // optional CLI args: pmin pmax steps
    double pmin = 0.5;
    double pmax = 1.0;
    int steps = 51;

    if (argc >= 4) {
        pmin = atof(argv[1]);
        pmax = atof(argv[2]);
        steps = atoi(argv[3]);
        if (steps < 1) steps = 51;
    }

    // prepare sample data: -1.0 .. 1.0 step 0.05
    vector<float> data;
    for (float x = -1.0f; x <= 1.0f + 1e-9f; x += 0.05f)
        data.push_back(x);

    // baseline
    vector<float> baseline_x;
    float mae_base = baseline_quant(data, baseline_x);
    cout << fixed << setprecision(6);
    cout << "Baseline MAE = " << mae_base << "\n\n";

    // grid search p
    double best_p = pmin;
    double best_mae = 1e30;
    vector<float> best_xhat;
    for (int i = 0; i < steps; ++i) {
        double p = pmin + (pmax - pmin) * (double)i / double(max(1, steps - 1));
        float scale = 0.0f;
        vector<float> xhat;
        float mae = pqnt::run(data, (float)p, xhat, scale);

        cout << "p=" << setw(6) << p
             << "    MAE=" << setw(12) << mae
             << "    scale=" << setw(10) << scale << "\n";

        if (mae < best_mae) {
            best_mae = mae;
            best_p = p;
            best_xhat = xhat;
        }
    }

    cout << "\nBest p = " << best_p << "    Best MAE = " << best_mae << "\n";

    return 0;
}


