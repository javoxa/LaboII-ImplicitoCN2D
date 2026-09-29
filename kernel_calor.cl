// kernel_calor.cl
// Kernels para Crank-Nicolson ADI 2D.

// RHS para la primera mitad del paso:
// b = (I + dt/2 * alfa * D_y^2) T + 0.5 * dt * Q
__kernel void calcular_b_y(
    __global const float* T,
    __global const float* Q,
    __global float* b,
    const int nx,
    const int ny,
    const float ry,
    const float dt)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;

    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) {
        b[idx] = 0.0f;
        return;
    }

    float T_arriba = T[(j + 1) * nx + i];
    float T_abajo  = T[(j - 1) * nx + i];

    b[idx] = (1.0f - 2.0f * ry) * T[idx]
             + ry * (T_arriba + T_abajo)
             + 0.5f * dt * Q[idx];
}

// RHS para la segunda mitad del paso:
// b = (I + dt/2 * alfa * D_x^2) T + 0.5 * dt * Q
__kernel void calcular_b_x(
    __global const float* T,
    __global const float* Q,
    __global float* b,
    const int nx,
    const int ny,
    const float rx,
    const float dt)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;

    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) {
        b[idx] = 0.0f;
        return;
    }

    float T_izq = T[j * nx + (i - 1)];
    float T_der = T[j * nx + (i + 1)];

    b[idx] = (1.0f - 2.0f * rx) * T[idx]
             + rx * (T_izq + T_der)
             + 0.5f * dt * Q[idx];
}

// Resuelve A_x T_half = b, donde
// A_x = I - dt/2 * alfa * D_x^2
// para cada fila interior j.
__kernel void solve_x(
    __global const float* b,
    __global float* T_half,
    __global float* cp,
    __global float* dp,
    const int nx,
    const int ny,
    const float rx)
{
    int j = get_global_id(0) + 1;

    if (j < 1 || j >= ny - 1) return;

    float a = 1.0f + 2.0f * rx;
    float c = -rx;        // superdiagonal
    float b_off = -rx;    // subdiagonal

    int m = nx - 2;       // puntos interiores en x: i = 1..nx-2
    int row_offset = j * nx;

    // Forward elimination
    int i = 1;
    int idx = row_offset + i;
    cp[row_offset] = c / a;
    dp[row_offset] = b[idx] / a;

    for (int l = 1; l < m; l++) {
        i = l + 1;
        idx = row_offset + i;

        float den = a - b_off * cp[row_offset + l - 1];
        cp[row_offset + l] = c / den;
        dp[row_offset + l] = (b[idx] - b_off * dp[row_offset + l - 1]) / den;
    }

    // Back substitution
    int l_last = m - 1;
    i = l_last + 1;
    idx = row_offset + i;
    T_half[idx] = dp[row_offset + l_last];

    for (int l = m - 2; l >= 0; l--) {
        i = l + 1;
        idx = row_offset + i;
        int idx_next = row_offset + (i + 1);

        T_half[idx] = dp[row_offset + l] - cp[row_offset + l] * T_half[idx_next];
    }
}

// Resuelve A_y T_new = b, donde
// A_y = I - dt/2 * alfa * D_y^2
// para cada columna interior i.
__kernel void solve_y(
    __global const float* b,
    __global float* T_new,
    __global float* cp,
    __global float* dp,
    const int nx,
    const int ny,
    const float ry)
{
    int i = get_global_id(0) + 1;

    if (i < 1 || i >= nx - 1) return;

    float a = 1.0f + 2.0f * ry;
    float c = -ry;
    float b_off = -ry;

    int m = ny - 2;       // puntos interiores en y: j = 1..ny-2

    // Forward elimination
    int j = 1;
    int idx = j * nx + i;
    cp[idx] = c / a;
    dp[idx] = b[idx] / a;

    for (int l = 1; l < m; l++) {
        j = l + 1;
        idx = j * nx + i;
        int idx_prev = (j - 1) * nx + i;

        float den = a - b_off * cp[idx_prev];
        cp[idx] = c / den;
        dp[idx] = (b[idx] - b_off * dp[idx_prev]) / den;
    }

    // Back substitution
    int l_last = m - 1;
    j = l_last + 1;
    idx = j * nx + i;
    T_new[idx] = dp[idx];

    for (int l = m - 2; l >= 0; l--) {
        j = l + 1;
        idx = j * nx + i;
        int idx_next = (j + 1) * nx + i;

        T_new[idx] = dp[idx] - cp[idx] * T_new[idx_next];
    }
}

// Impone condiciones Dirichlet en todo el borde.
__kernel void set_boundaries(
    __global float* T,
    const int nx,
    const int ny,
    const float T_frontera)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= nx || j >= ny) return;

    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        T[j * nx + i] = T_frontera;
    }
}
