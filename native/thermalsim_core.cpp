#include <cstddef>

#if defined(_WIN32)
#define THERMALSIM_EXPORT extern "C" __declspec(dllexport)
#else
#define THERMALSIM_EXPORT extern "C"
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

    const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(rows) * cols;
    const std::ptrdiff_t node_count = static_cast<std::ptrdiff_t>(layer_count) * plane;

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t index = 0; index < node_count; ++index) {
        const int layer = static_cast<int>(index / plane);
        const std::ptrdiff_t within_plane = index - static_cast<std::ptrdiff_t>(layer) * plane;
        const int row = static_cast<int>(within_plane / cols);
        const int col = static_cast<int>(within_plane - static_cast<std::ptrdiff_t>(row) * cols);
        const double centre = input[index];
        double value = boundary_diag[index] * centre;

        if (cols > 1) {
            const std::ptrdiff_t gx_base = (
                static_cast<std::ptrdiff_t>(layer) * rows + row
            ) * (cols - 1);
            if (col > 0) {
                value += gx[gx_base + col - 1] * (centre - input[index - 1]);
            }
            if (col + 1 < cols) {
                value += gx[gx_base + col] * (centre - input[index + 1]);
            }
        }
        if (rows > 1) {
            if (row > 0) {
                const std::ptrdiff_t gy_index = (
                    static_cast<std::ptrdiff_t>(layer) * (rows - 1) + row - 1
                ) * cols + col;
                value += gy[gy_index] * (centre - input[index - cols]);
            }
            if (row + 1 < rows) {
                const std::ptrdiff_t gy_index = (
                    static_cast<std::ptrdiff_t>(layer) * (rows - 1) + row
                ) * cols + col;
                value += gy[gy_index] * (centre - input[index + cols]);
            }
        }
        if (layer_count > 1) {
            if (layer > 0) {
                const std::ptrdiff_t gz_index = (
                    static_cast<std::ptrdiff_t>(layer - 1) * rows + row
                ) * cols + col;
                value += gz[gz_index] * (centre - input[index - plane]);
            }
            if (layer + 1 < layer_count) {
                const std::ptrdiff_t gz_index = (
                    static_cast<std::ptrdiff_t>(layer) * rows + row
                ) * cols + col;
                value += gz[gz_index] * (centre - input[index + plane]);
            }
        }
        output[index] = value;
    }
    return 0;
}
