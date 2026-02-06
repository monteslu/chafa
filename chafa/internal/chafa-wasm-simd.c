/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/* Copyright (C) 2018-2025 Hans Petter Jansson
 * Copyright (C) 2025-2026 Luis Montes - WASM SIMD implementation
 * Copyright (C) 2026 Radagast - MAXIMUM PERFORMANCE optimizations
 */

#include "config.h"

#ifdef HAVE_WASM_SIMD

#include <wasm_simd128.h>
#include <string.h>
#include "chafa.h"
#include "internal/chafa-private.h"

/* ============================================================================
 * MAXIMUM UNROLL: Process ALL 64 pixels with ZERO loop overhead
 * ============================================================================ */
gint
chafa_calc_cell_error_wasm_simd (const ChafaPixel *pixels, const ChafaColorPair *color_pair,
                                  const guint32 *sym_mask_u32)
{
    const guint32 fg_u32 = chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_FG]);
    const guint32 bg_u32 = chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_BG]);
    v128_t fg = wasm_i32x4_splat(fg_u32);
    v128_t bg = wasm_i32x4_splat(bg_u32);
    v128_t err = wasm_i32x4_splat(0);

    /* Macro for processing 4 pixels */
    #define PROCESS4(offset) do { \
        v128_t p = wasm_v128_load(&pixels[offset]); \
        v128_t m = wasm_v128_load(&sym_mask_u32[offset]); \
        v128_t s = wasm_v128_bitselect(fg, bg, m); \
        v128_t d = wasm_v128_or(wasm_u8x16_sub_sat(p, s), wasm_u8x16_sub_sat(s, p)); \
        v128_t lo = wasm_u16x8_extend_low_u8x16(d); \
        v128_t hi = wasm_u16x8_extend_high_u8x16(d); \
        err = wasm_i32x4_add(err, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(lo, lo))); \
        err = wasm_i32x4_add(err, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(hi, hi))); \
    } while(0)

    /* FULL UNROLL - all 64 pixels, zero loop overhead */
    PROCESS4(0);  PROCESS4(4);  PROCESS4(8);  PROCESS4(12);
    PROCESS4(16); PROCESS4(20); PROCESS4(24); PROCESS4(28);
    PROCESS4(32); PROCESS4(36); PROCESS4(40); PROCESS4(44);
    PROCESS4(48); PROCESS4(52); PROCESS4(56); PROCESS4(60);
    
    #undef PROCESS4

    /* Horizontal sum */
    err = wasm_i32x4_add(err, wasm_i32x4_shuffle(err, err, 2, 3, 0, 1));
    err = wasm_i32x4_add(err, wasm_i32x4_shuffle(err, err, 1, 0, 3, 2));
    return wasm_i32x4_extract_lane(err, 0);
}

/* ============================================================================
 * FULL UNROLL bitmap - 16 pixels per macro, 4 invocations
 * ============================================================================ */
guint64
chafa_work_cell_to_bitmap_wasm_simd (const ChafaPixel *pixels, const ChafaColorPair *color_pair)
{
    v128_t fg = wasm_i32x4_splat(chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_FG]));
    v128_t bg = wasm_i32x4_splat(chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_BG]));
    
    #define BITMAP8(offset) ({ \
        v128_t p0 = wasm_v128_load(&pixels[offset]); \
        v128_t p1 = wasm_v128_load(&pixels[offset + 4]); \
        v128_t abg0 = wasm_v128_or(wasm_u8x16_sub_sat(p0, bg), wasm_u8x16_sub_sat(bg, p0)); \
        v128_t abg1 = wasm_v128_or(wasm_u8x16_sub_sat(p1, bg), wasm_u8x16_sub_sat(bg, p1)); \
        v128_t afg0 = wasm_v128_or(wasm_u8x16_sub_sat(p0, fg), wasm_u8x16_sub_sat(fg, p0)); \
        v128_t afg1 = wasm_v128_or(wasm_u8x16_sub_sat(p1, fg), wasm_u8x16_sub_sat(fg, p1)); \
        v128_t dbg0 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(abg0)); \
        v128_t dfg0 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(afg0)); \
        v128_t dbg1 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(abg1)); \
        v128_t dfg1 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(afg1)); \
        ((wasm_i32x4_bitmask(wasm_i32x4_gt(dbg0, dfg0)) << 4) | wasm_i32x4_bitmask(wasm_i32x4_gt(dbg1, dfg1))); \
    })
    
    guint64 bitmap = ((guint64)BITMAP8(0) << 56) | ((guint64)BITMAP8(8) << 48) |
                     ((guint64)BITMAP8(16) << 40) | ((guint64)BITMAP8(24) << 32) |
                     ((guint64)BITMAP8(32) << 24) | ((guint64)BITMAP8(40) << 16) |
                     ((guint64)BITMAP8(48) << 8) | (guint64)BITMAP8(56);
    #undef BITMAP8
    
    return bitmap;
}

/* ============================================================================
 * FULL UNROLL mean colors - 16 pixels per block
 * ============================================================================ */
void
chafa_extract_cell_mean_colors_wasm_simd (const ChafaPixel *pixels, ChafaColorAccum *accums_out,
                                           const guint32 *sym_mask_u32)
{
    v128_t fg_acc = wasm_i32x4_splat(0);
    v128_t bg_acc = wasm_i32x4_splat(0);

    #define ACCUM4(offset) do { \
        v128_t p = wasm_v128_load(&pixels[offset]); \
        v128_t m = wasm_v128_load(&sym_mask_u32[offset]); \
        v128_t fp = wasm_v128_and(p, m); \
        v128_t bp = wasm_v128_andnot(p, m); \
        v128_t fl = wasm_u16x8_extend_low_u8x16(fp); \
        v128_t fh = wasm_u16x8_extend_high_u8x16(fp); \
        v128_t bl = wasm_u16x8_extend_low_u8x16(bp); \
        v128_t bh = wasm_u16x8_extend_high_u8x16(bp); \
        fg_acc = wasm_i32x4_add(fg_acc, wasm_u32x4_extend_low_u16x8(fl)); \
        fg_acc = wasm_i32x4_add(fg_acc, wasm_u32x4_extend_high_u16x8(fl)); \
        fg_acc = wasm_i32x4_add(fg_acc, wasm_u32x4_extend_low_u16x8(fh)); \
        fg_acc = wasm_i32x4_add(fg_acc, wasm_u32x4_extend_high_u16x8(fh)); \
        bg_acc = wasm_i32x4_add(bg_acc, wasm_u32x4_extend_low_u16x8(bl)); \
        bg_acc = wasm_i32x4_add(bg_acc, wasm_u32x4_extend_high_u16x8(bl)); \
        bg_acc = wasm_i32x4_add(bg_acc, wasm_u32x4_extend_low_u16x8(bh)); \
        bg_acc = wasm_i32x4_add(bg_acc, wasm_u32x4_extend_high_u16x8(bh)); \
    } while(0)

    /* FULL UNROLL */
    ACCUM4(0);  ACCUM4(4);  ACCUM4(8);  ACCUM4(12);
    ACCUM4(16); ACCUM4(20); ACCUM4(24); ACCUM4(28);
    ACCUM4(32); ACCUM4(36); ACCUM4(40); ACCUM4(44);
    ACCUM4(48); ACCUM4(52); ACCUM4(56); ACCUM4(60);
    #undef ACCUM4

    accums_out[0].ch[0] = wasm_i32x4_extract_lane(bg_acc, 0);
    accums_out[0].ch[1] = wasm_i32x4_extract_lane(bg_acc, 1);
    accums_out[0].ch[2] = wasm_i32x4_extract_lane(bg_acc, 2);
    accums_out[0].ch[3] = wasm_i32x4_extract_lane(bg_acc, 3);
    accums_out[1].ch[0] = wasm_i32x4_extract_lane(fg_acc, 0);
    accums_out[1].ch[1] = wasm_i32x4_extract_lane(fg_acc, 1);
    accums_out[1].ch[2] = wasm_i32x4_extract_lane(fg_acc, 2);
    accums_out[1].ch[3] = wasm_i32x4_extract_lane(fg_acc, 3);
}

/* Keep existing helper functions */
static const guint16 invdiv16[257] = {
    0, 32768, 16384, 10922, 8192, 6553, 5461, 4681, 4096, 3640, 3276,
    2978, 2730, 2520, 2340, 2184, 2048, 1927, 1820, 1724, 1638, 1560,
    1489, 1424, 1365, 1310, 1260, 1213, 1170, 1129, 1092, 1057, 1024,
    992, 963, 936, 910, 885, 862, 840, 819, 799, 780, 762, 744, 728,
    712, 697, 682, 668, 655, 642, 630, 618, 606, 595, 585, 574, 564,
    555, 546, 537, 528, 520, 512, 504, 496, 489, 481, 474, 468, 461,
    455, 448, 442, 436, 431, 425, 420, 414, 409, 404, 399, 394, 390,
    385, 381, 376, 372, 368, 364, 360, 356, 352, 348, 344, 341, 337,
    334, 330, 327, 324, 321, 318, 315, 312, 309, 306, 303, 300, 297,
    295, 292, 289, 287, 284, 282, 280, 277, 275, 273, 270, 268, 266,
    264, 262, 260, 258, 256, 254, 252, 250, 248, 246, 244, 242, 240,
    239, 237, 235, 234, 232, 230, 229, 227, 225, 224, 222, 221, 219,
    218, 217, 215, 214, 212, 211, 210, 208, 207, 206, 204, 203, 202,
    201, 199, 198, 197, 196, 195, 193, 192, 191, 190, 189, 188, 187,
    186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174,
    173, 172, 171, 170, 169, 168, 168, 167, 166, 165, 164, 163, 163,
    162, 161, 160, 159, 159, 158, 157, 156, 156, 155, 154, 153, 153,
    152, 151, 151, 150, 149, 148, 148, 147, 146, 146, 145, 144, 144,
    143, 143, 142, 141, 141, 140, 140, 139, 138, 138, 137, 137, 136,
    135, 135, 134, 134, 133, 133, 132, 132, 131, 131, 130, 130, 129,
    129, 128, 128
};

void
chafa_color_accum_div_scalar_wasm_simd (ChafaColorAccum *accum, guint16 divisor)
{
    v128_t acc = wasm_v128_load64_zero(accum);
    v128_t div = wasm_i16x8_splat(invdiv16[divisor]);
    v128_t plo = wasm_i32x4_extmul_low_i16x8(acc, div);
    v128_t phi = wasm_i32x4_extmul_high_i16x8(acc, div);
    v128_t rnd = wasm_i32x4_splat(0x4000);
    plo = wasm_i32x4_shr(wasm_i32x4_add(plo, rnd), 15);
    phi = wasm_i32x4_shr(wasm_i32x4_add(phi, rnd), 15);
    *(guint64*)accum = wasm_i64x2_extract_lane(wasm_i16x8_narrow_i32x4(plo, phi), 0);
}

gint
chafa_color_diff_4x_wasm_simd (const ChafaColor *target, const ChafaColor *palette, gint n_colors)
{
    v128_t min_dist = wasm_i32x4_splat(0x7FFFFFFF);
    v128_t min_idx = wasm_i32x4_splat(0);
    gint i;
    for (i = 0; i + 4 <= n_colors; i += 4) {
        gint32 d[4];
        for (int j = 0; j < 4; j++) {
            gint dr = palette[i+j].ch[0] - target->ch[0];
            gint dg = palette[i+j].ch[1] - target->ch[1];
            gint db = palette[i+j].ch[2] - target->ch[2];
            d[j] = dr*dr + dg*dg + db*db;
        }
        v128_t dv = wasm_v128_load(d);
        v128_t iv = wasm_i32x4_make(i, i+1, i+2, i+3);
        v128_t m = wasm_i32x4_lt(dv, min_dist);
        min_dist = wasm_v128_bitselect(dv, min_dist, m);
        min_idx = wasm_v128_bitselect(iv, min_idx, m);
    }
    gint32 d0=wasm_i32x4_extract_lane(min_dist,0), d1=wasm_i32x4_extract_lane(min_dist,1);
    gint32 d2=wasm_i32x4_extract_lane(min_dist,2), d3=wasm_i32x4_extract_lane(min_dist,3);
    gint bi=wasm_i32x4_extract_lane(min_idx,0); gint32 bd=d0;
    if(d1<bd){bd=d1;bi=wasm_i32x4_extract_lane(min_idx,1);}
    if(d2<bd){bd=d2;bi=wasm_i32x4_extract_lane(min_idx,2);}
    if(d3<bd){bd=d3;bi=wasm_i32x4_extract_lane(min_idx,3);}
    for(;i<n_colors;i++){
        gint dr=palette[i].ch[0]-target->ch[0];
        gint dg=palette[i].ch[1]-target->ch[1];
        gint db=palette[i].ch[2]-target->ch[2];
        gint32 d=dr*dr+dg*dg+db*db;
        if(d<bd){bd=d;bi=i;}
    }
    return bi;
}

#endif /* HAVE_WASM_SIMD */
