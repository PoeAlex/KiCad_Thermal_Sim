#include <cstddef>

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

THERMALSIM_EXPORT int thermalsim_core_version() {
    return 1;
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
        layer_count <= 0 || rows <= 0 || cols <= 0
        || boundary_diag == nullptr || input == nullptr || output == nullptr
    ) {
        return 1;
    }
    if (
        (cols > 1 && gx == nullptr)
        || (rows > 1 && gy == nullptr)
        || (layer_count > 1 && gz == nullptr)
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t node_count = n_layers * plane;
    const std::ptrdiff_t total_rows = n_layers * n_rows;
    const std::ptrdiff_t gx_stride = n_cols - 1;
    const std::ptrdiff_t gy_layer_stride = (n_rows - 1) * n_cols;

    #pragma omp parallel for schedule(static) if(node_count >= 16384)
    for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
        const std::ptrdiff_t layer = lr / n_rows;
        const std::ptrdiff_t row = lr - layer * n_rows;
        const std::ptrdiff_t row_offset = lr * n_cols;

        const double* THERMALSIM_RESTRICT u = input + row_offset;
        const double* THERMALSIM_RESTRICT b = boundary_diag + row_offset;
        double* THERMALSIM_RESTRICT out = output + row_offset;

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
    return 0;
}
