#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include "ramen.hpp"

void exp_naive(double *x, const double *y, int n) { for (int i = 0; i < n; i++) x[i] = std::exp(y[i]); }
void sigmoid_naive(double *x, const double *y, int n) { for (int i = 0; i < n; i++) x[i] = 1.0 / (1.0 + std::exp(-y[i])); }
void tanh_naive(double *x, const double *y, int n) { for (int i = 0; i < n; i++) x[i] = std::tanh(y[i]); }

void exp_ramen(double *dst, const double *src, int n) { Ramen::Exp<8, 4>::call(dst, src, n); }
void sigmoid_ramen(double *dst, const double *src, int n) { Ramen::Sigmoid<8, 4>::call(dst, src, n); }
void tanh_ramen(double *dst, const double *src, int n) { Ramen::Tanh<8, 4>::call(dst, src, n); }

struct Measure {
    template<typename F, typename ...Args>
    static inline auto invoke(F&& func, Args&&... args) {
        auto start = std::chrono::steady_clock::now();
        std::invoke(std::forward<decltype(func)>(func), std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        return end - start;
    }
};

template<class F>
struct TestFunc {
    std::string name;
    F func;
};
template<class F>
TestFunc(const char *, F) -> TestFunc<F>;

template<size_t Size>
struct TestSummary {
    double elapsed_time;
    double output[Size];
};

template<class F, size_t N>
struct TestGroup {
    std::string name;
    TestFunc<F> funcs[N];

    template<size_t Size, size_t NRepeat>
    void run() const {
        double input[Size]{};

        std::random_device sg;
        std::default_random_engine rng(sg());
        std::normal_distribution<double> dist(0, 1);

        for (auto& v: input) v = dist(rng);

        TestSummary<Size> summaries[N]{};

        for (auto i = 0; i < N; i++) {
            auto& func = funcs[i];
            auto& summary = summaries[i];
            for (auto j = 0; j < NRepeat; j++) {
                auto duration = Measure::invoke(func.func, summary.output, input, Size);
                summary.elapsed_time += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / 1e6;
            }
            summary.elapsed_time /= NRepeat;
        }

        std::printf("[%s]\n", name.c_str());
        for (auto i = 0; i < N; i++) {
            auto func = funcs[i];
            auto summary = summaries[i];
            if (i == 0) {
                std::printf("%10s(  c)\t%.6lf[ms]\n", func.name.c_str(), summary.elapsed_time);
            } else {
                auto control = summaries[0];

                double mae = 0.0, rmse = 0.0;
                for (auto j = 0; j < Size; j++) {
                    double eps = std::fabs(summary.output[j] - control.output[j]) / (control.output[j] + 1e-100);
                    rmse += eps * eps;
                    mae = std::max(mae, eps);
                }
                rmse = std::sqrt(rmse / Size);
                std::printf("%10s(t%02d)\t%.6lf[ms](%2.5lfx faster than c)\tmae=%.20e\trmse=%.20e\n", func.name.c_str(), i, summary.elapsed_time, control.elapsed_time / summary.elapsed_time , mae, rmse);
            }
        }
    }
};
template<class F, template<class> class... TF>
TestGroup(const char *, TF<F>...) -> TestGroup<F, sizeof...(TF)>;

struct Tester {
    template<size_t Size, size_t NRepeat = 10, class... TG>
    static void run(const TG&... tgs) {
        for (const auto& tg : {tgs...}) {
            tg.template run<Size, NRepeat>();
        }
    }
};

int main() {
    Tester::run<1000000>(
        TestGroup{
            "exp",
            TestFunc{"naive", exp_naive},
            TestFunc{"ramen", exp_ramen}
        },
        TestGroup{
            "sigmoid",
            TestFunc{"naive", sigmoid_naive},
            TestFunc{"ramen", sigmoid_ramen}
        },
        TestGroup{
            "tanh",
            TestFunc{"naive", tanh_naive},
            TestFunc{"ramen", tanh_ramen}
        }
    );
}
