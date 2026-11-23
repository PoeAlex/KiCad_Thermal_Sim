#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

template <typename T>
static inline bool has_valid_grid_and_conductances(
    int layer_count,
    int rows,
    int cols,
    const T* gx,
    const T* gy,
    const T* gz
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

static inline bool has_valid_compact_inputs(
    int layer_count,
    int rows,
    int cols,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut
) {
    if (layer_count <= 0 || rows <= 0 || cols <= 0) {
        return false;
    }
    if (
        copper_mask == nullptr
        || hs_mask == nullptr
        || cap_lut == nullptr
        || (cols > 1 && gx_lut == nullptr)
        || (rows > 1 && gy_lut == nullptr)
        || (layer_count > 1 && (via_enh == nullptr || gz_scale == nullptr))
    ) {
        return false;
    }
    return true;
}

template <typename InT, typename OutT>
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
    const InT* gx,
    const InT* gy,
    const InT* gz,
    const InT* boundary_diag,
    const InT* input,
    OutT* THERMALSIM_RESTRICT out
) {
    const std::ptrdiff_t row_offset = lr * n_cols;
    const InT* THERMALSIM_RESTRICT u = input + row_offset;
    const InT* THERMALSIM_RESTRICT b = boundary_diag + row_offset;

    // Boundary diagonal + X-direction coupling (branch-free interior)
    if (n_cols == 1) {
        out[0] = static_cast<OutT>(b[0]) * static_cast<OutT>(u[0]);
    } else {
        const InT* THERMALSIM_RESTRICT gx_row = gx + lr * gx_stride;
        out[0] = static_cast<OutT>(b[0]) * static_cast<OutT>(u[0])
            + static_cast<OutT>(gx_row[0]) * (static_cast<OutT>(u[0]) - static_cast<OutT>(u[1]));
        for (std::ptrdiff_t col = 1; col < n_cols - 1; ++col) {
            const OutT centre = static_cast<OutT>(u[col]);
            out[col] = static_cast<OutT>(b[col]) * centre
                + static_cast<OutT>(gx_row[col - 1]) * (centre - static_cast<OutT>(u[col - 1]))
                + static_cast<OutT>(gx_row[col]) * (centre - static_cast<OutT>(u[col + 1]));
        }
        const std::ptrdiff_t last = n_cols - 1;
        out[last] = static_cast<OutT>(b[last]) * static_cast<OutT>(u[last])
            + static_cast<OutT>(gx_row[last - 1])
                * (static_cast<OutT>(u[last]) - static_cast<OutT>(u[last - 1]));
    }

    // Y-direction coupling (branch-free over columns)
    if (n_rows > 1) {
        const bool has_up = row > 0;
        const bool has_down = row + 1 < n_rows;
        if (has_up && has_down) {
            const InT* THERMALSIM_RESTRICT gy_up = gy + layer * gy_layer_stride + (row - 1) * n_cols;
            const InT* THERMALSIM_RESTRICT gy_dn = gy_up + n_cols;
            const InT* THERMALSIM_RESTRICT u_up = u - n_cols;
            const InT* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const OutT centre = static_cast<OutT>(u[col]);
                out[col] += static_cast<OutT>(gy_up[col]) * (centre - static_cast<OutT>(u_up[col]))
                    + static_cast<OutT>(gy_dn[col]) * (centre - static_cast<OutT>(u_dn[col]));
            }
        } else if (has_up) {
            const InT* THERMALSIM_RESTRICT gy_up = gy + layer * gy_layer_stride + (row - 1) * n_cols;
            const InT* THERMALSIM_RESTRICT u_up = u - n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += static_cast<OutT>(gy_up[col])
                    * (static_cast<OutT>(u[col]) - static_cast<OutT>(u_up[col]));
            }
        } else if (has_down) {
            const InT* THERMALSIM_RESTRICT gy_dn = gy + layer * gy_layer_stride + row * n_cols;
            const InT* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += static_cast<OutT>(gy_dn[col])
                    * (static_cast<OutT>(u[col]) - static_cast<OutT>(u_dn[col]));
            }
        }
    }

    // Z-direction coupling (branch-free over columns)
    if (n_layers > 1) {
        const bool has_prev = layer > 0;
        const bool has_next = layer + 1 < n_layers;
        if (has_prev && has_next) {
            const InT* THERMALSIM_RESTRICT gz_prev = gz + ((layer - 1) * n_rows + row) * n_cols;
            const InT* THERMALSIM_RESTRICT gz_next = gz_prev + plane;
            const InT* THERMALSIM_RESTRICT u_prev = u - plane;
            const InT* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const OutT centre = static_cast<OutT>(u[col]);
                out[col] += static_cast<OutT>(gz_prev[col]) * (centre - static_cast<OutT>(u_prev[col]))
                    + static_cast<OutT>(gz_next[col]) * (centre - static_cast<OutT>(u_next[col]));
            }
        } else if (has_prev) {
            const InT* THERMALSIM_RESTRICT gz_prev = gz + ((layer - 1) * n_rows + row) * n_cols;
            const InT* THERMALSIM_RESTRICT u_prev = u - plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += static_cast<OutT>(gz_prev[col])
                    * (static_cast<OutT>(u[col]) - static_cast<OutT>(u_prev[col]));
            }
        } else if (has_next) {
            const InT* THERMALSIM_RESTRICT gz_next = gz + (layer * n_rows + row) * n_cols;
            const InT* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                out[col] += static_cast<OutT>(gz_next[col])
                    * (static_cast<OutT>(u[col]) - static_cast<OutT>(u_next[col]));
            }
        }
    }
}

template <typename VecT, bool ComputeDiag>
static inline void eval_compact_row(
    std::ptrdiff_t n_layers,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    std::ptrdiff_t plane,
    std::ptrdiff_t layer,
    std::ptrdiff_t row,
    std::ptrdiff_t lr,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    double scale,
    const VecT* input,
    double* THERMALSIM_RESTRICT ax_out,
    double* THERMALSIM_RESTRICT full_diag_out
) {
    const bool is_top = (layer == 0);
    const bool is_bot = (layer == n_layers - 1);
    const std::ptrdiff_t row_offset = lr * n_cols;
    const std::uint8_t* THERMALSIM_RESTRICT cm = copper_mask + row_offset;
    const std::uint8_t* THERMALSIM_RESTRICT hs = hs_mask + row * n_cols;
    const VecT* THERMALSIM_RESTRICT u = input + row_offset;
    const double* THERMALSIM_RESTRICT gx_l = gx_lut + layer * 3;
    const double* THERMALSIM_RESTRICT gy_l = gy_lut + layer * 3;
    const double* THERMALSIM_RESTRICT cap_l = cap_lut + layer * 2;
    const double top_term = is_top ? top_diag : 0.0;

    auto cell_diag_mass = [&](std::ptrdiff_t col, int m_c) -> double {
        double b_diag = top_term;
        double cell_cap = cap_l[m_c];
        if (is_bot) {
            const bool is_hs = (hs[col] != 0);
            b_diag += is_hs ? bot_pad_diag : bot_air_diag;
            if (is_hs) {
                cell_cap += pad_cap_cell;
            }
        }
        return b_diag + scale * cell_cap;
    };

    if (n_cols == 1) {
        const int m_c = cm[0] != 0 ? 1 : 0;
        const double diag_mass = cell_diag_mass(0, m_c);
        ax_out[0] = diag_mass * static_cast<double>(u[0]);
        if (ComputeDiag) {
            full_diag_out[0] = diag_mass;
        }
    } else {
        int m_curr = cm[0] != 0 ? 1 : 0;
        int m_next = cm[1] != 0 ? 1 : 0;
        double g_right = gx_l[m_curr + m_next];
        {
            const double diag_mass = cell_diag_mass(0, m_curr);
            const double u0 = static_cast<double>(u[0]);
            const double u1 = static_cast<double>(u[1]);
            ax_out[0] = diag_mass * u0 + g_right * (u0 - u1);
            if (ComputeDiag) {
                full_diag_out[0] = diag_mass + g_right;
            }
        }
        for (std::ptrdiff_t col = 1; col < n_cols - 1; ++col) {
            const double g_left = g_right;
            m_curr = m_next;
            m_next = cm[col + 1] != 0 ? 1 : 0;
            g_right = gx_l[m_curr + m_next];
            const double diag_mass = cell_diag_mass(col, m_curr);
            const double u_c = static_cast<double>(u[col]);
            ax_out[col] = diag_mass * u_c
                + g_left * (u_c - static_cast<double>(u[col - 1]))
                + g_right * (u_c - static_cast<double>(u[col + 1]));
            if (ComputeDiag) {
                full_diag_out[col] = diag_mass + g_left + g_right;
            }
        }
        {
            const std::ptrdiff_t last = n_cols - 1;
            const double g_left = g_right;
            const double diag_mass = cell_diag_mass(last, m_next);
            const double u_last = static_cast<double>(u[last]);
            ax_out[last] = diag_mass * u_last
                + g_left * (u_last - static_cast<double>(u[last - 1]));
            if (ComputeDiag) {
                full_diag_out[last] = diag_mass + g_left;
            }
        }
    }

    if (n_rows > 1) {
        const bool has_up = row > 0;
        const bool has_down = row + 1 < n_rows;
        if (has_up && has_down) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_up = cm - n_cols;
            const std::uint8_t* THERMALSIM_RESTRICT cm_dn = cm + n_cols;
            const VecT* THERMALSIM_RESTRICT u_up = u - n_cols;
            const VecT* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                const double g_u = gy_l[m_c + (cm_up[col] != 0 ? 1 : 0)];
                const double g_d = gy_l[m_c + (cm_dn[col] != 0 ? 1 : 0)];
                const double u_c = static_cast<double>(u[col]);
                ax_out[col] += g_u * (u_c - static_cast<double>(u_up[col]))
                    + g_d * (u_c - static_cast<double>(u_dn[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_u + g_d;
                }
            }
        } else if (has_up) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_up = cm - n_cols;
            const VecT* THERMALSIM_RESTRICT u_up = u - n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                const double g_u = gy_l[m_c + (cm_up[col] != 0 ? 1 : 0)];
                ax_out[col] += g_u * (static_cast<double>(u[col]) - static_cast<double>(u_up[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_u;
                }
            }
        } else if (has_down) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_dn = cm + n_cols;
            const VecT* THERMALSIM_RESTRICT u_dn = u + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                const double g_d = gy_l[m_c + (cm_dn[col] != 0 ? 1 : 0)];
                ax_out[col] += g_d * (static_cast<double>(u[col]) - static_cast<double>(u_dn[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_d;
                }
            }
        }
    }

    if (n_layers > 1) {
        const float* THERMALSIM_RESTRICT ve_row = via_enh + row * n_cols;
        const bool has_prev = layer > 0;
        const bool has_next = layer + 1 < n_layers;
        if (has_prev && has_next) {
            const double s_p = gz_scale[layer - 1];
            const double s_n = gz_scale[layer];
            const VecT* THERMALSIM_RESTRICT u_prev = u - plane;
            const VecT* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const double ve = static_cast<double>(ve_row[col]);
                const double g_p = s_p * ve;
                const double g_n = s_n * ve;
                const double u_c = static_cast<double>(u[col]);
                ax_out[col] += g_p * (u_c - static_cast<double>(u_prev[col]))
                    + g_n * (u_c - static_cast<double>(u_next[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_p + g_n;
                }
            }
        } else if (has_prev) {
            const double s_p = gz_scale[layer - 1];
            const VecT* THERMALSIM_RESTRICT u_prev = u - plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const double g_p = s_p * static_cast<double>(ve_row[col]);
                ax_out[col] += g_p * (static_cast<double>(u[col]) - static_cast<double>(u_prev[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_p;
                }
            }
        } else if (has_next) {
            const double s_n = gz_scale[layer];
            const VecT* THERMALSIM_RESTRICT u_next = u + plane;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const double g_n = s_n * static_cast<double>(ve_row[col]);
                ax_out[col] += g_n * (static_cast<double>(u[col]) - static_cast<double>(u_next[col]));
                if (ComputeDiag) {
                    full_diag_out[col] += g_n;
                }
            }
        }
    }
}

static inline void compute_compact_full_diag_row(
    std::ptrdiff_t n_layers,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    std::ptrdiff_t layer,
    std::ptrdiff_t row,
    std::ptrdiff_t lr,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    double scale,
    double* THERMALSIM_RESTRICT full_diag_out
) {
    const bool is_top = (layer == 0);
    const bool is_bot = (layer == n_layers - 1);
    const std::ptrdiff_t row_offset = lr * n_cols;
    const std::uint8_t* THERMALSIM_RESTRICT cm = copper_mask + row_offset;
    const std::uint8_t* THERMALSIM_RESTRICT hs = hs_mask + row * n_cols;
    const double* THERMALSIM_RESTRICT gx_l = gx_lut + layer * 3;
    const double* THERMALSIM_RESTRICT gy_l = gy_lut + layer * 3;
    const double* THERMALSIM_RESTRICT cap_l = cap_lut + layer * 2;
    const double top_term = is_top ? top_diag : 0.0;

    auto cell_diag_mass = [&](std::ptrdiff_t col, int m_c) -> double {
        double b_diag = top_term;
        double cell_cap = cap_l[m_c];
        if (is_bot) {
            const bool is_hs = (hs[col] != 0);
            b_diag += is_hs ? bot_pad_diag : bot_air_diag;
            if (is_hs) {
                cell_cap += pad_cap_cell;
            }
        }
        return b_diag + scale * cell_cap;
    };

    if (n_cols == 1) {
        const int m_c = cm[0] != 0 ? 1 : 0;
        full_diag_out[0] = cell_diag_mass(0, m_c);
    } else {
        int m_curr = cm[0] != 0 ? 1 : 0;
        int m_next = cm[1] != 0 ? 1 : 0;
        double g_right = gx_l[m_curr + m_next];
        full_diag_out[0] = cell_diag_mass(0, m_curr) + g_right;
        for (std::ptrdiff_t col = 1; col < n_cols - 1; ++col) {
            const double g_left = g_right;
            m_curr = m_next;
            m_next = cm[col + 1] != 0 ? 1 : 0;
            g_right = gx_l[m_curr + m_next];
            full_diag_out[col] = cell_diag_mass(col, m_curr) + g_left + g_right;
        }
        full_diag_out[n_cols - 1] = cell_diag_mass(n_cols - 1, m_next) + g_right;
    }

    if (n_rows > 1) {
        const bool has_up = row > 0;
        const bool has_down = row + 1 < n_rows;
        if (has_up && has_down) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_up = cm - n_cols;
            const std::uint8_t* THERMALSIM_RESTRICT cm_dn = cm + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                full_diag_out[col] += gy_l[m_c + (cm_up[col] != 0 ? 1 : 0)]
                    + gy_l[m_c + (cm_dn[col] != 0 ? 1 : 0)];
            }
        } else if (has_up) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_up = cm - n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                full_diag_out[col] += gy_l[m_c + (cm_up[col] != 0 ? 1 : 0)];
            }
        } else if (has_down) {
            const std::uint8_t* THERMALSIM_RESTRICT cm_dn = cm + n_cols;
            for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                const int m_c = cm[col] != 0 ? 1 : 0;
                full_diag_out[col] += gy_l[m_c + (cm_dn[col] != 0 ? 1 : 0)];
            }
        }
    }

    if (n_layers > 1) {
        const float* THERMALSIM_RESTRICT ve_row = via_enh + row * n_cols;
        const double z_factor = (layer > 0 ? gz_scale[layer - 1] : 0.0)
            + (layer + 1 < n_layers ? gz_scale[layer] : 0.0);
        for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
            full_diag_out[col] += z_factor * static_cast<double>(ve_row[col]);
        }
    }
}

template <typename T>
static inline int prolong_add_2x2_impl(
    int layer_count,
    int rows,
    int cols,
    const T* coarse_x,
    T* fine_x
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

        const T* THERMALSIM_RESTRICT src = coarse_x + (layer * coarse_rows + cr) * coarse_cols;
        T* THERMALSIM_RESTRICT dst = fine_x + lr * n_cols;

        for (std::ptrdiff_t cc = 0; cc < full_coarse_cols; ++cc) {
            const T val = src[cc];
            dst[2 * cc] += val;
            dst[2 * cc + 1] += val;
        }
        if (has_odd_col) {
            dst[n_cols - 1] += src[full_coarse_cols];
        }
    }
    return 0;
}

}  // namespace

THERMALSIM_EXPORT int thermalsim_core_version() {
    return 3;
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
    return prolong_add_2x2_impl<double>(layer_count, rows, cols, coarse_x, fine_x);
}

THERMALSIM_EXPORT int thermalsim_apply_compact(
    int layer_count,
    int rows,
    int cols,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    double scale,
    const double* input,
    double* output
) {
    if (
        !has_valid_compact_inputs(
            layer_count, rows, cols, copper_mask, via_enh, hs_mask, gx_lut, gy_lut, gz_scale, cap_lut
        )
        || input == nullptr || output == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t total_rows = n_layers * n_rows;

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
    for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
        const std::ptrdiff_t layer = lr / n_rows;
        const std::ptrdiff_t row = lr - layer * n_rows;
        eval_compact_row<double, false>(
            n_layers,
            n_rows,
            n_cols,
            plane,
            layer,
            row,
            lr,
            copper_mask,
            via_enh,
            hs_mask,
            gx_lut,
            gy_lut,
            gz_scale,
            cap_lut,
            pad_cap_cell,
            top_diag,
            bot_air_diag,
            bot_pad_diag,
            scale,
            input,
            output + lr * n_cols,
            nullptr
        );
    }
    return 0;
}

THERMALSIM_EXPORT int thermalsim_damped_jacobi_compact_f32(
    int layer_count,
    int rows,
    int cols,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    double scale,
    double omega,
    const float* rhs,
    float* x,
    float* scratch,
    int iterations,
    int zero_init
) {
    if (
        !has_valid_compact_inputs(
            layer_count, rows, cols, copper_mask, via_enh, hs_mask, gx_lut, gy_lut, gz_scale, cap_lut
        )
        || rhs == nullptr || x == nullptr || scratch == nullptr || iterations < 0
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
    const std::ptrdiff_t total_rows = n_layers * n_rows;

    std::atomic<int> alloc_failed{0};

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
#endif
    {
        constexpr std::ptrdiff_t kStackCols = 1024;
        double stack_ax[kStackCols];
        double stack_diag[kStackCols];
        double* heap_buf = nullptr;
        double* ax_buf = stack_ax;
        double* diag_buf = stack_diag;
        bool buf_ok = true;

        if (n_cols > kStackCols && iterations > 0) {
            heap_buf = new (std::nothrow) double[static_cast<std::size_t>(2 * n_cols)];
            if (heap_buf != nullptr) {
                ax_buf = heap_buf;
                diag_buf = heap_buf + n_cols;
            } else {
                buf_ok = false;
                alloc_failed.store(1, std::memory_order_relaxed);
            }
        }

        int remaining = iterations;
        if (zero_init != 0) {
            if (remaining == 0) {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    float* THERMALSIM_RESTRICT x_row = x + lr * n_cols;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        x_row[col] = 0.0f;
                    }
                }
            } else {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    if (!buf_ok) {
                        continue;
                    }
                    const std::ptrdiff_t layer = lr / n_rows;
                    const std::ptrdiff_t row = lr - layer * n_rows;
                    compute_compact_full_diag_row(
                        n_layers,
                        n_rows,
                        n_cols,
                        layer,
                        row,
                        lr,
                        copper_mask,
                        via_enh,
                        hs_mask,
                        gx_lut,
                        gy_lut,
                        gz_scale,
                        cap_lut,
                        pad_cap_cell,
                        top_diag,
                        bot_air_diag,
                        bot_pad_diag,
                        scale,
                        diag_buf
                    );
                    const std::ptrdiff_t row_offset = lr * n_cols;
                    float* THERMALSIM_RESTRICT x_row = x + row_offset;
                    const float* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        const double fd = diag_buf[col];
                        const double w_inv = (std::fabs(fd) > 1e-30) ? (omega / fd) : 0.0;
                        x_row[col] = static_cast<float>(w_inv * static_cast<double>(r_row[col]));
                    }
                }
                remaining -= 1;
            }
        }

        float* src = x;
        float* dst = scratch;
        for (int iter = 0; iter < remaining; ++iter) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                if (!buf_ok) {
                    continue;
                }
                const std::ptrdiff_t layer = lr / n_rows;
                const std::ptrdiff_t row = lr - layer * n_rows;
                eval_compact_row<float, true>(
                    n_layers,
                    n_rows,
                    n_cols,
                    plane,
                    layer,
                    row,
                    lr,
                    copper_mask,
                    via_enh,
                    hs_mask,
                    gx_lut,
                    gy_lut,
                    gz_scale,
                    cap_lut,
                    pad_cap_cell,
                    top_diag,
                    bot_air_diag,
                    bot_pad_diag,
                    scale,
                    src,
                    ax_buf,
                    diag_buf
                );
                const std::ptrdiff_t row_offset = lr * n_cols;
                const float* THERMALSIM_RESTRICT src_row = src + row_offset;
                const float* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                float* THERMALSIM_RESTRICT dst_row = dst + row_offset;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    const double fd = diag_buf[col];
                    const double w_inv = (std::fabs(fd) > 1e-30) ? (omega / fd) : 0.0;
                    dst_row[col] = static_cast<float>(
                        static_cast<double>(src_row[col])
                        + w_inv * (static_cast<double>(r_row[col]) - ax_buf[col])
                    );
                }
            }
            float* tmp = src;
            src = dst;
            dst = tmp;
        }

        if (src != x) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                const std::ptrdiff_t row_offset = lr * n_cols;
                float* THERMALSIM_RESTRICT x_row = x + row_offset;
                const float* THERMALSIM_RESTRICT s_row = src + row_offset;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    x_row[col] = s_row[col];
                }
            }
        }

        delete[] heap_buf;
    }

    return alloc_failed.load(std::memory_order_relaxed) != 0 ? 1 : 0;
}

THERMALSIM_EXPORT int thermalsim_residual_restrict_2x2_compact_f32(
    int layer_count,
    int rows,
    int cols,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    double scale,
    const float* rhs,
    const float* x,
    float* coarse_rhs
) {
    if (
        !has_valid_compact_inputs(
            layer_count, rows, cols, copper_mask, via_enh, hs_mask, gx_lut, gy_lut, gz_scale, cap_lut
        )
        || rhs == nullptr || x == nullptr || coarse_rhs == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t plane = n_rows * n_cols;
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

            eval_compact_row<float, false>(
                n_layers,
                n_rows,
                n_cols,
                plane,
                layer,
                r0,
                lr0,
                copper_mask,
                via_enh,
                hs_mask,
                gx_lut,
                gy_lut,
                gz_scale,
                cap_lut,
                pad_cap_cell,
                top_diag,
                bot_air_diag,
                bot_pad_diag,
                scale,
                x,
                row0,
                nullptr
            );

            const float* THERMALSIM_RESTRICT rhs0 = rhs + lr0 * n_cols;
            double* THERMALSIM_RESTRICT sum_row = row0;

            if (r1 < n_rows) {
                const std::ptrdiff_t lr1 = lr0 + 1;
                eval_compact_row<float, false>(
                    n_layers,
                    n_rows,
                    n_cols,
                    plane,
                    layer,
                    r1,
                    lr1,
                    copper_mask,
                    via_enh,
                    hs_mask,
                    gx_lut,
                    gy_lut,
                    gz_scale,
                    cap_lut,
                    pad_cap_cell,
                    top_diag,
                    bot_air_diag,
                    bot_pad_diag,
                    scale,
                    x,
                    row1,
                    nullptr
                );
                const float* THERMALSIM_RESTRICT rhs1 = rhs + lr1 * n_cols;
                const double* THERMALSIM_RESTRICT ax1 = row1;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = (static_cast<double>(rhs0[col]) - sum_row[col])
                        + (static_cast<double>(rhs1[col]) - ax1[col]);
                }
            } else {
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = static_cast<double>(rhs0[col]) - sum_row[col];
                }
            }

            float* THERMALSIM_RESTRICT dst = coarse_rhs + lcr * coarse_cols;
            for (std::ptrdiff_t cc = 0; cc < full_coarse_cols; ++cc) {
                dst[cc] = static_cast<float>(sum_row[2 * cc] + sum_row[2 * cc + 1]);
            }
            if (has_odd_col) {
                dst[full_coarse_cols] = static_cast<float>(sum_row[n_cols - 1]);
            }
        }

        delete[] heap_buf;
    }

    return alloc_failed.load(std::memory_order_relaxed) != 0 ? 1 : 0;
}

THERMALSIM_EXPORT int thermalsim_coarsen_compact_to_f32(
    int layer_count,
    int rows,
    int cols,
    const std::uint8_t* copper_mask,
    const float* via_enh,
    const std::uint8_t* hs_mask,
    const double* gx_lut,
    const double* gy_lut,
    const double* gz_scale,
    const double* cap_lut,
    double pad_cap_cell,
    double top_diag,
    double bot_air_diag,
    double bot_pad_diag,
    float* gx_c,
    float* gy_c,
    float* gz_c,
    float* boundary_c,
    float* stiffness_diag_c,
    float* cap_c
) {
    if (
        !has_valid_compact_inputs(
            layer_count, rows, cols, copper_mask, via_enh, hs_mask, gx_lut, gy_lut, gz_scale, cap_lut
        )
        || boundary_c == nullptr || stiffness_diag_c == nullptr || cap_c == nullptr
    ) {
        return 1;
    }

    const std::ptrdiff_t n_layers = static_cast<std::ptrdiff_t>(layer_count);
    const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(rows);
    const std::ptrdiff_t n_cols = static_cast<std::ptrdiff_t>(cols);
    const std::ptrdiff_t coarse_rows = (n_rows + 1) / 2;
    const std::ptrdiff_t coarse_cols = (n_cols + 1) / 2;
    const std::ptrdiff_t total_coarse_rows = n_layers * coarse_rows;

    if (
        (coarse_cols > 1 && gx_c == nullptr)
        || (coarse_rows > 1 && gy_c == nullptr)
        || (n_layers > 1 && gz_c == nullptr)
    ) {
        return 1;
    }

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * n_rows * n_cols);
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
#endif
    {
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t lcr = 0; lcr < total_coarse_rows; ++lcr) {
            const std::ptrdiff_t layer = lcr / coarse_rows;
            const std::ptrdiff_t cr = lcr - layer * coarse_rows;
            const std::ptrdiff_t r0 = cr * 2;
            const std::ptrdiff_t r1 = r0 + 1;
            const bool has_r1 = r1 < n_rows;

            const bool is_top = (layer == 0);
            const bool is_bot = (layer == n_layers - 1);
            const double top_term = is_top ? top_diag : 0.0;

            const std::uint8_t* THERMALSIM_RESTRICT cm0 = copper_mask + (layer * n_rows + r0) * n_cols;
            const std::uint8_t* THERMALSIM_RESTRICT cm1 = has_r1 ? (cm0 + n_cols) : nullptr;
            const std::uint8_t* THERMALSIM_RESTRICT hs0 = hs_mask + r0 * n_cols;
            const std::uint8_t* THERMALSIM_RESTRICT hs1 = has_r1 ? (hs0 + n_cols) : nullptr;
            const float* THERMALSIM_RESTRICT ve0 = (n_layers > 1) ? (via_enh + r0 * n_cols) : nullptr;
            const float* THERMALSIM_RESTRICT ve1 = (n_layers > 1 && has_r1) ? (ve0 + n_cols) : nullptr;

            const double* THERMALSIM_RESTRICT gx_l = gx_lut + layer * 3;
            const double* THERMALSIM_RESTRICT gy_l = gy_lut + layer * 3;
            const double* THERMALSIM_RESTRICT cap_l = cap_lut + layer * 2;

            const bool do_gy = (coarse_rows > 1) && (cr + 1 < coarse_rows);
            const std::uint8_t* THERMALSIM_RESTRICT cm_r1 = do_gy ? (copper_mask + (layer * n_rows + r1) * n_cols) : nullptr;
            const std::uint8_t* THERMALSIM_RESTRICT cm_r2 = do_gy ? (cm_r1 + n_cols) : nullptr;
            const bool do_gz = (n_layers > 1) && (layer + 1 < n_layers);
            const double gz_s = do_gz ? gz_scale[layer] : 0.0;

            float* THERMALSIM_RESTRICT b_dst = boundary_c + lcr * coarse_cols;
            float* THERMALSIM_RESTRICT c_dst = cap_c + lcr * coarse_cols;
            float* THERMALSIM_RESTRICT gx_dst = (coarse_cols > 1) ? (gx_c + lcr * (coarse_cols - 1)) : nullptr;
            float* THERMALSIM_RESTRICT gy_dst = do_gy
                ? (gy_c + (layer * (coarse_rows - 1) + cr) * coarse_cols)
                : nullptr;
            float* THERMALSIM_RESTRICT gz_dst = do_gz
                ? (gz_c + (layer * coarse_rows + cr) * coarse_cols)
                : nullptr;

            for (std::ptrdiff_t cc = 0; cc < coarse_cols; ++cc) {
                const std::ptrdiff_t c0 = cc * 2;
                const std::ptrdiff_t c1 = c0 + 1;
                const bool has_c1 = c1 < n_cols;

                const int m00 = cm0[c0] != 0 ? 1 : 0;
                double b_sum = top_term;
                double cap_sum = cap_l[m00];
                if (is_bot) {
                    const bool is_hs = (hs0[c0] != 0);
                    b_sum += is_hs ? bot_pad_diag : bot_air_diag;
                    if (is_hs) {
                        cap_sum += pad_cap_cell;
                    }
                }
                if (has_c1) {
                    const int m01 = cm0[c1] != 0 ? 1 : 0;
                    b_sum += top_term;
                    cap_sum += cap_l[m01];
                    if (is_bot) {
                        const bool is_hs = (hs0[c1] != 0);
                        b_sum += is_hs ? bot_pad_diag : bot_air_diag;
                        if (is_hs) {
                            cap_sum += pad_cap_cell;
                        }
                    }
                }
                if (has_r1) {
                    const int m10 = cm1[c0] != 0 ? 1 : 0;
                    b_sum += top_term;
                    cap_sum += cap_l[m10];
                    if (is_bot) {
                        const bool is_hs = (hs1[c0] != 0);
                        b_sum += is_hs ? bot_pad_diag : bot_air_diag;
                        if (is_hs) {
                            cap_sum += pad_cap_cell;
                        }
                    }
                    if (has_c1) {
                        const int m11 = cm1[c1] != 0 ? 1 : 0;
                        b_sum += top_term;
                        cap_sum += cap_l[m11];
                        if (is_bot) {
                            const bool is_hs = (hs1[c1] != 0);
                            b_sum += is_hs ? bot_pad_diag : bot_air_diag;
                            if (is_hs) {
                                cap_sum += pad_cap_cell;
                            }
                        }
                    }
                }
                b_dst[cc] = static_cast<float>(b_sum);
                c_dst[cc] = static_cast<float>(cap_sum);

                if (coarse_cols > 1 && cc + 1 < coarse_cols) {
                    const std::ptrdiff_t c_left = c1;
                    const std::ptrdiff_t c_right = c1 + 1;
                    double gx_sum = gx_l[(cm0[c_left] != 0 ? 1 : 0) + (cm0[c_right] != 0 ? 1 : 0)];
                    if (has_r1) {
                        gx_sum += gx_l[(cm1[c_left] != 0 ? 1 : 0) + (cm1[c_right] != 0 ? 1 : 0)];
                    }
                    gx_dst[cc] = static_cast<float>(gx_sum);
                }

                if (do_gy) {
                    double gy_sum = gy_l[(cm_r1[c0] != 0 ? 1 : 0) + (cm_r2[c0] != 0 ? 1 : 0)];
                    if (has_c1) {
                        gy_sum += gy_l[(cm_r1[c1] != 0 ? 1 : 0) + (cm_r2[c1] != 0 ? 1 : 0)];
                    }
                    gy_dst[cc] = static_cast<float>(gy_sum);
                }

                if (do_gz) {
                    double ve_sum = static_cast<double>(ve0[c0]);
                    if (has_c1) {
                        ve_sum += static_cast<double>(ve0[c1]);
                    }
                    if (has_r1) {
                        ve_sum += static_cast<double>(ve1[c0]);
                        if (has_c1) {
                            ve_sum += static_cast<double>(ve1[c1]);
                        }
                    }
                    gz_dst[cc] = static_cast<float>(gz_s * ve_sum);
                }
            }
        }

#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t lcr = 0; lcr < total_coarse_rows; ++lcr) {
            const std::ptrdiff_t layer = lcr / coarse_rows;
            const std::ptrdiff_t cr = lcr - layer * coarse_rows;
            const float* THERMALSIM_RESTRICT b_row = boundary_c + lcr * coarse_cols;
            float* THERMALSIM_RESTRICT diag_row = stiffness_diag_c + lcr * coarse_cols;

            const float* THERMALSIM_RESTRICT gx_row = (coarse_cols > 1)
                ? (gx_c + lcr * (coarse_cols - 1))
                : nullptr;
            const float* THERMALSIM_RESTRICT gy_up = (coarse_rows > 1 && cr > 0)
                ? (gy_c + (layer * (coarse_rows - 1) + (cr - 1)) * coarse_cols)
                : nullptr;
            const float* THERMALSIM_RESTRICT gy_dn = (coarse_rows > 1 && cr + 1 < coarse_rows)
                ? (gy_c + (layer * (coarse_rows - 1) + cr) * coarse_cols)
                : nullptr;
            const float* THERMALSIM_RESTRICT gz_prev = (n_layers > 1 && layer > 0)
                ? (gz_c + ((layer - 1) * coarse_rows + cr) * coarse_cols)
                : nullptr;
            const float* THERMALSIM_RESTRICT gz_next = (n_layers > 1 && layer + 1 < n_layers)
                ? (gz_c + (layer * coarse_rows + cr) * coarse_cols)
                : nullptr;

            for (std::ptrdiff_t cc = 0; cc < coarse_cols; ++cc) {
                double d = static_cast<double>(b_row[cc]);
                if (gx_row != nullptr) {
                    if (cc > 0) {
                        d += static_cast<double>(gx_row[cc - 1]);
                    }
                    if (cc + 1 < coarse_cols) {
                        d += static_cast<double>(gx_row[cc]);
                    }
                }
                if (gy_up != nullptr) {
                    d += static_cast<double>(gy_up[cc]);
                }
                if (gy_dn != nullptr) {
                    d += static_cast<double>(gy_dn[cc]);
                }
                if (gz_prev != nullptr) {
                    d += static_cast<double>(gz_prev[cc]);
                }
                if (gz_next != nullptr) {
                    d += static_cast<double>(gz_next[cc]);
                }
                diag_row[cc] = static_cast<float>(d);
            }
        }
    }

    return 0;
}

THERMALSIM_EXPORT int thermalsim_damped_jacobi_f32(
    int layer_count,
    int rows,
    int cols,
    const float* gx,
    const float* gy,
    const float* gz,
    const float* diag_mass,
    const float* omega_inv_diag,
    const float* rhs,
    float* x,
    float* scratch,
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

    std::atomic<int> alloc_failed{0};

#if defined(_OPENMP)
    const int n_threads = choose_num_threads(n_layers * plane);
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
#endif
    {
        constexpr std::ptrdiff_t kStackCols = 1024;
        double stack_ax[kStackCols];
        double* heap_buf = nullptr;
        double* ax_buf = stack_ax;
        bool buf_ok = true;

        if (n_cols > kStackCols && iterations > 0) {
            heap_buf = new (std::nothrow) double[static_cast<std::size_t>(n_cols)];
            if (heap_buf != nullptr) {
                ax_buf = heap_buf;
            } else {
                buf_ok = false;
                alloc_failed.store(1, std::memory_order_relaxed);
            }
        }

        int remaining = iterations;
        if (zero_init != 0) {
            if (remaining == 0) {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    float* THERMALSIM_RESTRICT x_row = x + lr * n_cols;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        x_row[col] = 0.0f;
                    }
                }
            } else {
#if defined(_OPENMP)
                #pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                    const std::ptrdiff_t row_offset = lr * n_cols;
                    float* THERMALSIM_RESTRICT x_row = x + row_offset;
                    const float* THERMALSIM_RESTRICT w_row = omega_inv_diag + row_offset;
                    const float* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                    for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                        x_row[col] = static_cast<float>(
                            static_cast<double>(w_row[col]) * static_cast<double>(r_row[col])
                        );
                    }
                }
                remaining -= 1;
            }
        }

        float* src = x;
        float* dst = scratch;
        for (int iter = 0; iter < remaining; ++iter) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                if (!buf_ok) {
                    continue;
                }
                const std::ptrdiff_t layer = lr / n_rows;
                const std::ptrdiff_t row = lr - layer * n_rows;
                apply_structured_row<float, double>(
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
                    src,
                    ax_buf
                );
                const std::ptrdiff_t row_offset = lr * n_cols;
                const float* THERMALSIM_RESTRICT src_row = src + row_offset;
                const float* THERMALSIM_RESTRICT w_row = omega_inv_diag + row_offset;
                const float* THERMALSIM_RESTRICT r_row = rhs + row_offset;
                float* THERMALSIM_RESTRICT dst_row = dst + row_offset;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    dst_row[col] = static_cast<float>(
                        static_cast<double>(src_row[col])
                        + static_cast<double>(w_row[col])
                            * (static_cast<double>(r_row[col]) - ax_buf[col])
                    );
                }
            }
            float* tmp = src;
            src = dst;
            dst = tmp;
        }

        if (src != x) {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t lr = 0; lr < total_rows; ++lr) {
                const std::ptrdiff_t row_offset = lr * n_cols;
                float* THERMALSIM_RESTRICT x_row = x + row_offset;
                const float* THERMALSIM_RESTRICT s_row = src + row_offset;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    x_row[col] = s_row[col];
                }
            }
        }

        delete[] heap_buf;
    }

    return alloc_failed.load(std::memory_order_relaxed) != 0 ? 1 : 0;
}

THERMALSIM_EXPORT int thermalsim_residual_restrict_2x2_f32(
    int layer_count,
    int rows,
    int cols,
    const float* gx,
    const float* gy,
    const float* gz,
    const float* diag_mass,
    const float* rhs,
    const float* x,
    float* coarse_rhs
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

            apply_structured_row<float, double>(
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

            const float* THERMALSIM_RESTRICT rhs0 = rhs + lr0 * n_cols;
            double* THERMALSIM_RESTRICT sum_row = row0;

            if (r1 < n_rows) {
                const std::ptrdiff_t lr1 = lr0 + 1;
                apply_structured_row<float, double>(
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
                const float* THERMALSIM_RESTRICT rhs1 = rhs + lr1 * n_cols;
                const double* THERMALSIM_RESTRICT ax1 = row1;
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = (static_cast<double>(rhs0[col]) - sum_row[col])
                        + (static_cast<double>(rhs1[col]) - ax1[col]);
                }
            } else {
                for (std::ptrdiff_t col = 0; col < n_cols; ++col) {
                    sum_row[col] = static_cast<double>(rhs0[col]) - sum_row[col];
                }
            }

            float* THERMALSIM_RESTRICT dst = coarse_rhs + lcr * coarse_cols;
            for (std::ptrdiff_t cc = 0; cc < full_coarse_cols; ++cc) {
                dst[cc] = static_cast<float>(sum_row[2 * cc] + sum_row[2 * cc + 1]);
            }
            if (has_odd_col) {
                dst[full_coarse_cols] = static_cast<float>(sum_row[n_cols - 1]);
            }
        }

        delete[] heap_buf;
    }

    return alloc_failed.load(std::memory_order_relaxed) != 0 ? 1 : 0;
}

THERMALSIM_EXPORT int thermalsim_prolong_add_2x2_f32(
    int layer_count,
    int rows,
    int cols,
    const float* coarse_x,
    float* fine_x
) {
    return prolong_add_2x2_impl<float>(layer_count, rows, cols, coarse_x, fine_x);
}
