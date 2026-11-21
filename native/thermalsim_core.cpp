#include <atomic>
#include <cstddef>
#include <new>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#define THERMALSIM_EXPORT extern "C" __declspec(dllexport)
#else
#define THERMALSIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
#define THERMALSIM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define THERMALSIM_RESTRICT __restrict__
#else
#define THERMALSIM_RESTRICT
#endif

namespace {

#if defined(_OPENMP)
static inline int choose_num_threads(std::ptrdiff_t node_count) {
    if (node_count < 32768) {
        return 1;
    }
    const std::ptrdiff_t work_threads = node_count / 32768;
    int cap = work_threads >= 4 ? 4 : static_cast<int>(work_threads);
    const int max_threads = omp_get_max_threads();
    if (max_threads < cap) {
        cap = max_threads;
    }
    return cap > 1 ? cap : 1;
}
#endif

static inline bool has_valid_grid_and_conductances(
    int layer_count,
    int rows,
    int cols,
    const double* gx,
    const double* gy,
    const double* gz
) {
    if (layer_count <= 0 || rows <= 0 || cols <= 0) {
        return false;
    }
    if (
        (cols > 1 && gx == nullptr)
        || (rows > 1 && gy == nullptr)
        || (layer_count > 1 && gz == nullptr)
    ) {
        return false;
    }
    return true;
}

static inline void apply_structured_row(
    std::ptrdiff_t n_layers,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    std::ptrdiff_t plane,
    std::ptrdiff_t gx_stride,
    std::ptrdiff_t gy_layer_stride,
    std::ptrdiff_t layer,
    std::ptrdiff_t row,
    std::ptrdiff_t lr,
    const double* gx,
    const double* gy,
    const double* gz,
    const double* boundary_diag,
    const double* input,
    double* THERMALSIM_RESTRICT out
) {
    const std::ptrdiff_t row_offset = lr * n_cols;
    const double* THERMALSIM_RESTRICT u = input + row_offset;
    const double* THERMALSIM_RESTRICT b = boundary_diag + row_offset;

    // Boundary diagonal + X-direction coupling (branch-free interior)
    if (n_cols == 1) {
        out[0] = b[0] * u[0];
    } else {
        const double* THERMALSIM_RESTRICT gx_row = gx + lr * gx_stride;
        out[0] = b[0] * u[0] + gx_row[0] * (u[0] - u[1]);
        for (std::ptrdiff_t col = 1; col < n_cols - 1; ++col) {
            const double centre = u[col];
            out[col] = b[col] * centre
                + gx_row[col - 1] * (centre - u[col - 1])
                + gx_row[col] * (centre - u[col + 1]);
        }
        const std::ptrdiff_t last = n_cols - 1;
        out[last] = b[last] * u[last] + gx_row[last - 1] * (u[last] - u[last - 1]);
    }

    // Y-direction coupling (branch-free over columns)
    if (n_rows > 1) {
        const bool has_up = row > 0;
        const bool has_down = row + 1 < n_rows;
        if (has_up && has_down) {
            const double* THERMALSIM_RESTRICT gy_up = gy + layer * gy_layer_stride + (row - 1) * n_cols;
            const double* THERMALSIM_RESTRICT gy_dn = gy_up + n_cols;
            const double* THERMALSIM_RESTRICT u_up = u - n_cols;
            const double* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const double centre = u[col];
                out[col] += gy_up[col] * (centre - u_up[col])
                    + gy_dn[col] * (centre - u_dn[col]);
            }
        } else if (has_up) {
            const double* THERMALSIM_RESTRICT gy_up = gy + layer * gy_layer_stride + (row - 1) * n_cols;
            const double* THERMALSIM_RESTRICT u_up = u - n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += gy_up[col] * (u[col] - u_up[col]);
            }
        } else if (has_down) {
            const double* THERMALSIM_RESTRICT gy_dn = gy + layer * gy_layer_stride + row * n_cols;
            const double* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += gy_dn[col] * (u[col] - u_dn[col]);
            }
        }
    }

    // Z-direction coupling (branch-free over columns)
    if (n_layers > 1) {
        const bool has_prev = layer > 0;
        const bool has_next = layer + 1 < n_layers;
        if (has_prev && has_next) {
            const double* THERMALSIM_RESTRICT gz_prev = gz + ((layer - 1) * n_rows + row) * n_cols;
            const double* THERMALSIM_RESTRICT gz_next = gz_prev + plane;
            const double* THERMALSIM_RESTRICT u_prev = u - plane;
            const double* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const double centre = u[col];
                out[col] += gz_prev[col] * (centre - u_prev[col])
                    + gz_next[col] * (centre - u_next[col]);
            }
        } else if (has_prev) {
            const double* THERMALSIM_RESTRICT gz_prev = gz + ((layer - 1) * n_rows + row) * n_cols;
            const double* THERMALSIM_RESTRICT u_prev = u - plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += gz_prev[col] * (u[col] - u_prev[col]);
            }
        } else if (has_next) {
            const double* THERMALSIM_RESTRICT gz_next = gz + (layer * n_rows + row) * n_cols;
            const double* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += gz_next[col] * (u[col] - u_next[col]);
            }
        }
    }
}

}  // namespace

THERMALSIM_EXPORT int thermalsim_core_version() {
    return 2;
}

THERMALSIM_EXPORT int thermalsim_apply_structured(
    int layer_count,
    int rows,
    int cols,
    const double* gx,
    const double* gy,
    const double* gz,
    const double* boundary_diag,
    const double* input,
    double* output
) {
    if (
        !has_valid_grid_and_conductances(layer_count, rows, cols, gx, gy, gz)
        || boundary_diag == nullptr || input == nullptr || output == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t total_rows = n_layers * n_rows;
    const std::ptrdiff_t gx_stride = n_cols - 1;
    const std::ptrdiff_t gy_layer_stride = (n_rows - 1) * n_cols;

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
    for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
        const std::ptrdiff_t layer = lr / n_rows;
        const std::ptrdiff_t row = lr - layer * n_rows;
        apply_structured_row(
            n_layers,
            n_rows,
            n_cols,
            plane,
            gx_stride,
            gy_layer_stride,
            layer,
            row,
            lr,
            gx,
            gy,
            gz,
            boundary_diag,
            input,
            output + lr * n_cols
        );
    }
    return 0;
}

THERMALSIM_EXPORT int thermalsim_damped_jacobi(
    int layer_count,
    int rows,
    int cols,
    const double* gx,
    const double* gy,
    const double* gz,
    const double* diag_mass,
    const double* omega_inv_diag,
    const double* rhs,
    double* x,
    double* scratch,
    int iterations,
    int zero_init
) {
    if (
        !has_valid_grid_and_conductances(layer_count, rows, cols, gx, gy, gz)
        || diag_mass == nullptr || omega_inv_diag == nullptr
        || rhs == nullptr || x == nullptr || scratch == nullptr
        || iterations < 0
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t total_rows = n_layers * n_rows;
    const std::ptrdiff_t gx_stride = n_cols - 1;
    const std::ptrdiff_t gy_layer_stride = (n_rows - 1) * n_cols;

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
#endif
    {
        int remaining = iterations;
        if (zero_init != 0) {
            if (remaining == 0) {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    double* THERMALSIM_RESTRICT x_row = x + lr * n_cols;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        x_row[col] = 0.0;
                    }
                }
            } else {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    const std::ptrdiff_t row_offset = lr * n_cols;
                    double* THERMALSIM_RESTRICT x_row = x + row_offset;
                    const double* THERMALSIM_RESTRICT w_row = omega_inv_diag + row_offset;
                    const double* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        x_row[col] = w_row[col] * r_row[col];
                    }
                }
                remaining -= 1;
            }
        }

        for (int iter = 0; iter < remaining; ++iter) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                const std::ptrdiff_t layer = lr / n_rows;
                const std::ptrdiff_t row = lr - layer * n_rows;
                apply_structured_row(
                    n_layers,
                    n_rows,
                    n_cols,
                    plane,
                    gx_stride,
                    gy_layer_stride,
                    layer,
                    row,
                    lr,
                    gx,
                    gy,
                    gz,
                    diag_mass,
                    x,
                    scratch + lr * n_cols
                );
            }

#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                const std::ptrdiff_t row_offset = lr * n_cols;
                double* THERMALSIM_RESTRICT x_row = x + row_offset;
                const double* THERMALSIM_RESTRICT w_row = omega_inv_diag + row_offset;
                const double* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                const double* THERMALSIM_RESTRICT s_row = scratch + row_offset;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    x_row[col] += w_row[col] * (r_row[col] - s_row[col]);
                }
            }
        }
    }
    return 0;
}

THERMALSIM_EXPORT int thermalsim_residual_restrict_2x2(
    int layer_count,
    int rows,
    int cols,
    const double* gx,
    const double* gy,
    const double* gz,
    const double* diag_mass,
    const double* rhs,
    const double* x,
    double* coarse_rhs
) {
    if (
        !has_valid_grid_and_conductances(layer_count, rows, cols, gx, gy, gz)
        || diag_mass == nullptr || rhs == nullptr || x == nullptr || coarse_rhs == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t gx_stride = n_cols - 1;
    const std::ptrdiff_t gy_layer_stride = (n_rows - 1) * n_cols;

    const std::ptrdiff_t coarse_rows = (n_rows + 1) / 2;
    const std::ptrdiff_t coarse_cols = (n_cols + 1) / 2;
    const std::ptrdiff_t full_coarse_cols = n_cols / 2;
    const std::ptrdiff_t total_coarse_rows = n_layers * coarse_rows;
    const bool has_odd_col = (n_cols & 1) != 0;

    std::atomic<int> alloc_failed{0};

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
#endif
    {
        constexpr std::ptrdiff_t kStackCols = 1024;
        double stack_row0[kStackCols];
        double stack_row1[kStackCols];
        double* heap_buf = nullptr;
        double* row0 = stack_row0;
        double* row1 = stack_row1;
        bool buf_ok = true;

        if (n_cols > kStackCols) {
            heap_buf = new (std::nothrow) double[static_cast<std::size_t>(2 * n_cols)];
            if (heap_buf != nullptr) {
                row0 = heap_buf;
                row1 = heap_buf + n_cols;
            } else {
                buf_ok = false;
                alloc_failed.store(1, std::memory_order_relaxed);
            }
        }

#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t lcr = 0; lcr < total_coarse_rows; ++lcr) {
            if (!buf_ok) {
                continue;
            }
            const std::ptrdiff_t layer = lcr / coarse_rows;
            const std::ptrdiff_t cr = lcr - layer * coarse_rows;
            const std::ptrdiff_t r0 = cr * 2;
            const std::ptrdiff_t r1 = r0 + 1;
            const std::ptrdiff_t lr0 = layer * n_rows + r0;

            apply_structured_row(
                n_layers,
                n_rows,
                n_cols,
                plane,
                gx_stride,
                gy_layer_stride,
                layer,
                r0,
                lr0,
                gx,
                gy,
                gz,
                diag_mass,
                x,
                row0
            );

            const double* THERMALSIM_RESTRICT rhs0 = rhs + lr0 * n_cols;
            double* THERMALSIM_RESTRICT sum_row = row0;

            if (r1 < n_rows) {
                const std::ptrdiff_t lr1 = lr0 + 1;
                apply_structured_row(
                    n_layers,
                    n_rows,
                    n_cols,
                    plane,
                    gx_stride,
                    gy_layer_stride,
                    layer,
                    r1,
                    lr1,
                    gx,
                    gy,
                    gz,
                    diag_mass,
                    x,
                    row1
                );
                const double* THERMALSIM_RESTRICT rhs1 = rhs + lr1 * n_cols;
                const double* THERMALSIM_RESTRICT ax1 = row1;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = (rhs0[col] - sum_row[col]) + (rhs1[col] - ax1[col]);
                }
            } else {
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = rhs0[col] - sum_row[col];
                }
            }

            double* THERMALSIM_RESTRICT dst = coarse_rhs + lcr * coarse_cols;
            for (std::ptrdiff_t cc = 0; cc < full_coarse_cols; ++cc) {
                dst[cc] = sum_row[2 * cc] + sum_row[2 * cc + 1];
            }
            if (has_odd_col) {
                dst[full_coarse_cols] = sum_row[n_cols - 1];
            }
        }

        delete[] heap_buf;
    }

    return alloc_failed.load(std::memory_order_relaxed) != 0 ? 1 : 0;
}

THERMALSIM_EXPORT int thermalsim_prolong_add_2x2(
    int layer_count,
    int rows,
    int cols,
    const double* coarse_x,
    double* fine_x
) {
    if (
        layer_count <= 0 || rows <= 0 || cols <= 0
        || coarse_x == nullptr || fine_x == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t total_rows = n_layers * n_rows;
    const std::ptrdiff_t coarse_rows = (n_rows + 1) / 2;
    const std::ptrdiff_t coarse_cols = (n_cols + 1) / 2;
    const std::ptrdiff_t full_coarse_cols = n_cols / 2;
    const bool has_odd_col = (n_cols & 1) != 0;

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * n_rows * n_cols);
    #pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
    for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
        const std::ptrdiff_t layer = lr / n_rows;
        const std::ptrdiff_t row = lr - layer * n_rows;
        const std::ptrdiff_t cr = row >> 1;

        const double* THERMALSIM_RESTRICT src = coarse_x + (layer * coarse_rows + cr) * coarse_cols;
        double* THERMALSIM_RESTRICT dst = fine_x + lr * n_cols;

        for (std::ptrdiff_t cc = 0; cc < full_coarse_cols; ++cc) {
            const double val = src[cc];
            dst[2 * cc] += val;
            dst[2 * cc + 1] += val;
        }
        if (has_odd_col) {
            dst[n_cols - 1] += src[full_coarse_cols];
        }
    }
    return 0;
}
