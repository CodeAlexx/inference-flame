// Reference value generator using llama.cpp's canonical dequant kernels.
// Builds hand-crafted blocks, invokes dequantize_row_*, prints output as
// Rust-formatted constants.
//
// Build (see build.sh).

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Keep it simple — only need C-mode declarations.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-quants.h"

static uint16_t f32_to_f16_bits(float x) {
    return ggml_fp32_to_fp16(x);
}

static void print_floats(const char *label, const float *y, int count) {
    printf("/// %s\n", label);
    printf("const %s: [f32; %d] = [\n", label, count);
    for (int i = 0; i < count; i++) {
        if (i % 8 == 0) printf("    ");
        printf("%.7ef32", y[i]);
        if (i != count - 1) printf(",");
        if ((i + 1) % 8 == 0) printf("\n");
        else printf(" ");
    }
    printf("];\n\n");
}

static void print_bytes(const char *label, const void *data, int count) {
    const uint8_t *p = (const uint8_t *)data;
    printf("/// %s — raw block bytes\n", label);
    printf("const %s: [u8; %d] = [\n", label, count);
    for (int i = 0; i < count; i++) {
        if (i % 16 == 0) printf("    ");
        printf("0x%02X", p[i]);
        if (i != count - 1) printf(",");
        if ((i + 1) % 16 == 0) printf("\n");
        else printf(" ");
    }
    printf("];\n\n");
}

// ---------------------------------------------------------------------------
// Q8_0
// ---------------------------------------------------------------------------
static void gen_q8_0(void) {
    block_q8_0 b;
    b.d = f32_to_f16_bits(2.0f);
    for (int i = 0; i < 32; i++) b.qs[i] = (int8_t)(-16 + i);
    float y[32];
    dequantize_row_q8_0(&b, y, 32);
    print_bytes("Q8_0_BLOCK_BASIC", &b, sizeof(b));
    print_floats("Q8_0_REF_BASIC", y, 32);
}

// Pack 8 x 6-bit (scale, min) pairs into 12 bytes (ggml canonical layout).
static void pack_scale_min(uint8_t scales[12], const uint8_t sc[8], const uint8_t m[8]) {
    memset(scales, 0, 12);
    for (int j = 0; j < 4; j++) {
        scales[j]     = sc[j] & 0x3F;
        scales[j + 4] = m[j]  & 0x3F;
    }
    for (int j = 4; j < 8; j++) {
        uint8_t sc_lo = sc[j] & 0x0F;
        uint8_t sc_hi = (sc[j] >> 4) & 0x03;
        uint8_t m_lo  = m[j]  & 0x0F;
        uint8_t m_hi  = (m[j]  >> 4) & 0x03;
        scales[j + 4] = (m_lo << 4) | sc_lo;
        scales[j - 4] = (scales[j - 4] & 0x3F) | (sc_hi << 6);
        scales[j]     = (scales[j]     & 0x3F) | (m_hi  << 6);
    }
}

// ---------------------------------------------------------------------------
// Q4_K with ALL 8 sub-blocks non-zero + distinct, exercising j<4 AND j>=4.
// ---------------------------------------------------------------------------
static void gen_q4_k(void) {
    block_q4_K b;
    memset(&b, 0, sizeof(b));
    // The union has `ggml_half d, dmin` as the first struct member.
    // In C-mode (GGML_COMMON_AGGR_S/U empty), the anonymous struct access is
    // `b.d`, `b.dmin`. Confirmed by the C reference dequant using `x[i].d`.
    b.d    = f32_to_f16_bits(1.0f);
    b.dmin = f32_to_f16_bits(0.5f);

    uint8_t sc[8] = {3, 5, 7, 11, 13, 17, 19, 23};
    uint8_t m [8] = {1, 2, 4,  8, 16, 32, 33, 48};
    pack_scale_min(b.scales, sc, m);
    for (int i = 0; i < 128; i++) b.qs[i] = 0x3B;

    float y[256];
    dequantize_row_q4_K(&b, y, 256);
    print_bytes("Q4_K_BLOCK_DISTINCT_SCALES", &b, sizeof(b));
    print_floats("Q4_K_REF_DISTINCT_SCALES", y, 256);
}

// ---------------------------------------------------------------------------
// Q5_K — cases covering qh high-bit extraction progression.
// ---------------------------------------------------------------------------
static void gen_q5_k(void) {
    block_q5_K b;
    memset(&b, 0, sizeof(b));
    b.d    = f32_to_f16_bits(1.0f);
    b.dmin = f32_to_f16_bits(0.0f);

    uint8_t sc[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint8_t m [8] = {0, 0, 0, 0, 0, 0, 0, 0};
    pack_scale_min(b.scales, sc, m);
    for (int i = 0; i < 32; i++) b.qh[i] = 0xFF;
    for (int i = 0; i < 128; i++) b.qs[i] = 0x00;

    float y[256];
    dequantize_row_q5_K(&b, y, 256);
    print_bytes("Q5_K_BLOCK_ALL_HIGH_BITS", &b, sizeof(b));
    print_floats("Q5_K_REF_ALL_HIGH_BITS", y, 256);

    // Alternating qh pattern validates u1,u2 <<= 2 progression.
    for (int i = 0; i < 32; i++) b.qh[i] = 0xAA;
    dequantize_row_q5_K(&b, y, 256);
    print_bytes("Q5_K_BLOCK_ALTERNATING_QH", &b, sizeof(b));
    print_floats("Q5_K_REF_ALTERNATING_QH", y, 256);
}

// ---------------------------------------------------------------------------
// Q6_K — exercise all four shift positions of the 2-bit qh extraction.
// ---------------------------------------------------------------------------
static void gen_q6_k(void) {
    block_q6_K b;
    memset(&b, 0, sizeof(b));
    for (int i = 0; i < 16; i++) b.scales[i] = 1;
    for (int i = 0; i < 128; i++) b.ql[i] = 0x00;
    for (int i = 0; i < 64; i++)  b.qh[i] = 0xE4; // 0b11100100 → shifts 0,2,4,6 extract 0,1,2,3
    b.d = f32_to_f16_bits(1.0f);

    float y[256];
    dequantize_row_q6_K(&b, y, 256);
    print_bytes("Q6_K_BLOCK_FOUR_SHIFTS", &b, sizeof(b));
    print_floats("Q6_K_REF_FOUR_SHIFTS", y, 256);
}

int main(void) {
    printf("// AUTO-GENERATED by tools/gguf_ref/gen_ref.c using llama.cpp's\n");
    printf("// canonical dequantize_row_* kernels. Do not hand-edit.\n\n");
    gen_q8_0();
    gen_q4_k();
    gen_q5_k();
    gen_q6_k();
    return 0;
}
