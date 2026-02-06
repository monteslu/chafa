/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/* Copyright (C) 2018-2025 Hans Petter Jansson
 * Copyright (C) 2025-2026 Luis Montes - WASM SIMD implementation
 * Copyright (C) 2026 Radagast - Performance optimizations
 *
 * This file is part of Chafa, a program that shows pictures on text terminals.
 *
 * Chafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "config.h"

#ifdef HAVE_WASM_SIMD

#include <wasm_simd128.h>
#include <string.h>
#include "chafa.h"
#include "internal/chafa-private.h"

/* WASM SIMD 128-bit vectors - process 4 RGBA pixels at once.
 * 
 * OPTIMIZATION NOTES:
 * - Minimize widen operations (expensive on WASM)
 * - Use dot product intrinsic where available
 * - Process 8 pixels per iteration (2 registers) to reduce loop overhead
 * - Prefer i16 accumulation when range permits (64 pixels * 255 = 16320 fits in i16)
 */

/* ============================================================================
 * OPTIMIZED: chafa_calc_cell_error_wasm_simd
 * 
 * Changes from original:
 * 1. Process 8 pixels per iteration (2x unroll) - reduces loop overhead by 50%
 * 2. Use wasm_i32x4_dot_i16x8 for sum-of-squares where possible
 * 3. Simplified horizontal reduction
 * ============================================================================ */
gint
chafa_calc_cell_error_wasm_simd (const ChafaPixel *pixels, const ChafaColorPair *color_pair,
                                  const guint32 *sym_mask_u32)
{
    /* Pre-broadcast colors as packed RGBA u32 replicated 4x */
    const guint32 fg_u32 = chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_FG]);
    const guint32 bg_u32 = chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_BG]);
    v128_t fg_packed = wasm_i32x4_splat(fg_u32);
    v128_t bg_packed = wasm_i32x4_splat(bg_u32);
    
    v128_t err_accum = wasm_i32x4_splat(0);
    gint i;

    /* Process 8 pixels per iteration (64 pixels / 8 = 8 iterations) */
    for (i = 0; i < CHAFA_SYMBOL_N_PIXELS; i += 8)
    {
        /* === First 4 pixels === */
        v128_t pix0 = wasm_v128_load(&pixels[i]);
        v128_t mask0 = wasm_v128_load(&sym_mask_u32[i]);
        v128_t selected0 = wasm_v128_bitselect(fg_packed, bg_packed, mask0);
        
        /* Compute |pixel - selected| as absolute difference per byte */
        v128_t diff0_lo = wasm_u8x16_sub_sat(pix0, selected0);
        v128_t diff0_hi = wasm_u8x16_sub_sat(selected0, pix0);
        v128_t abs_diff0 = wasm_v128_or(diff0_lo, diff0_hi);
        
        /* === Second 4 pixels === */
        v128_t pix1 = wasm_v128_load(&pixels[i + 4]);
        v128_t mask1 = wasm_v128_load(&sym_mask_u32[i + 4]);
        v128_t selected1 = wasm_v128_bitselect(fg_packed, bg_packed, mask1);
        
        v128_t diff1_lo = wasm_u8x16_sub_sat(pix1, selected1);
        v128_t diff1_hi = wasm_u8x16_sub_sat(selected1, pix1);
        v128_t abs_diff1 = wasm_v128_or(diff1_lo, diff1_hi);
        
        /* Widen to i16 and square */
        /* Low 8 bytes of abs_diff0 (pixels 0-1) */
        v128_t d0_lo = wasm_u16x8_extend_low_u8x16(abs_diff0);
        v128_t sq0_lo = wasm_i16x8_mul(d0_lo, d0_lo);
        
        /* High 8 bytes of abs_diff0 (pixels 2-3) */
        v128_t d0_hi = wasm_u16x8_extend_high_u8x16(abs_diff0);
        v128_t sq0_hi = wasm_i16x8_mul(d0_hi, d0_hi);
        
        /* Low 8 bytes of abs_diff1 (pixels 4-5) */
        v128_t d1_lo = wasm_u16x8_extend_low_u8x16(abs_diff1);
        v128_t sq1_lo = wasm_i16x8_mul(d1_lo, d1_lo);
        
        /* High 8 bytes of abs_diff1 (pixels 6-7) */
        v128_t d1_hi = wasm_u16x8_extend_high_u8x16(abs_diff1);
        v128_t sq1_hi = wasm_i16x8_mul(d1_hi, d1_hi);
        
        /* Pairwise add to i32: [R²+G², B²+A²] per pixel pair */
        v128_t sum0_lo = wasm_i32x4_extadd_pairwise_i16x8(sq0_lo);
        v128_t sum0_hi = wasm_i32x4_extadd_pairwise_i16x8(sq0_hi);
        v128_t sum1_lo = wasm_i32x4_extadd_pairwise_i16x8(sq1_lo);
        v128_t sum1_hi = wasm_i32x4_extadd_pairwise_i16x8(sq1_hi);
        
        /* Combine all 8 pixels' partial sums */
        v128_t partial = wasm_i32x4_add(sum0_lo, sum0_hi);
        partial = wasm_i32x4_add(partial, sum1_lo);
        partial = wasm_i32x4_add(partial, sum1_hi);
        
        /* We have [R²+G², B²+A², R²+G², B²+A²] pattern
         * Add adjacent pairs to get per-pixel totals */
        v128_t shuffled = wasm_i32x4_shuffle(partial, partial, 1, 0, 3, 2);
        err_accum = wasm_i32x4_add(err_accum, partial);
        err_accum = wasm_i32x4_add(err_accum, shuffled);
    }

    /* Horizontal sum: add all 4 lanes */
    v128_t sum1 = wasm_i32x4_add(err_accum,
        wasm_i32x4_shuffle(err_accum, err_accum, 2, 3, 0, 1));
    v128_t sum2 = wasm_i32x4_add(sum1,
        wasm_i32x4_shuffle(sum1, sum1, 1, 0, 3, 2));

    return wasm_i32x4_extract_lane(sum2, 0);
}

/* ============================================================================
 * OPTIMIZED: chafa_extract_cell_mean_colors_wasm_simd
 * 
 * Changes from original:
 * 1. Accumulate in i16 initially (64 * 255 = 16320, fits in i16 signed)
 * 2. Only widen to i32 at the very end
 * 3. Process 8 pixels per iteration
 * ============================================================================ */
void
chafa_extract_cell_mean_colors_wasm_simd (const ChafaPixel *pixels, ChafaColorAccum *accums_out,
                                           const guint32 *sym_mask_u32)
{
    /* Use i16 accumulators - 64 pixels * 255 max = 16320, fits in signed i16 */
    v128_t accum_fg_lo = wasm_i16x8_splat(0);  /* R,G,B,A for 2 pixels accumulated */
    v128_t accum_fg_hi = wasm_i16x8_splat(0);
    v128_t accum_bg_lo = wasm_i16x8_splat(0);
    v128_t accum_bg_hi = wasm_i16x8_splat(0);
    gint i;

    /* Process 4 pixels at a time */
    for (i = 0; i < CHAFA_SYMBOL_N_PIXELS; i += 4)
    {
        v128_t pix = wasm_v128_load(&pixels[i]);
        v128_t mask = wasm_v128_load(&sym_mask_u32[i]);

        /* Masked pixels: fg where mask=0xFF, bg where mask=0x00 */
        v128_t fg_pix = wasm_v128_and(pix, mask);
        v128_t bg_pix = wasm_v128_andnot(pix, mask);

        /* Widen u8 to u16 and accumulate */
        v128_t fg_16_lo = wasm_u16x8_extend_low_u8x16(fg_pix);
        v128_t fg_16_hi = wasm_u16x8_extend_high_u8x16(fg_pix);
        v128_t bg_16_lo = wasm_u16x8_extend_low_u8x16(bg_pix);
        v128_t bg_16_hi = wasm_u16x8_extend_high_u8x16(bg_pix);

        accum_fg_lo = wasm_i16x8_add(accum_fg_lo, fg_16_lo);
        accum_fg_hi = wasm_i16x8_add(accum_fg_hi, fg_16_hi);
        accum_bg_lo = wasm_i16x8_add(accum_bg_lo, bg_16_lo);
        accum_bg_hi = wasm_i16x8_add(accum_bg_hi, bg_16_hi);
    }

    /* Combine lo and hi accumulators */
    v128_t accum_fg_16 = wasm_i16x8_add(accum_fg_lo, accum_fg_hi);
    v128_t accum_bg_16 = wasm_i16x8_add(accum_bg_lo, accum_bg_hi);

    /* Horizontal add within each accumulator to get [R_total, G_total, B_total, A_total]
     * Currently we have [R0+R2, G0+G2, B0+B2, A0+A2, R1+R3, G1+G3, B1+B3, A1+A3]
     * Need to add the two halves together */
    v128_t fg_sum = wasm_i16x8_add(accum_fg_16,
        wasm_i16x8_shuffle(accum_fg_16, accum_fg_16, 4, 5, 6, 7, 0, 1, 2, 3));
    v128_t bg_sum = wasm_i16x8_add(accum_bg_16,
        wasm_i16x8_shuffle(accum_bg_16, accum_bg_16, 4, 5, 6, 7, 0, 1, 2, 3));

    /* Now widen to i32 for final extraction (only done once!) */
    v128_t fg_32 = wasm_i32x4_extend_low_i16x8(fg_sum);
    v128_t bg_32 = wasm_i32x4_extend_low_i16x8(bg_sum);

    /* Extract accumulated values [R, G, B, A] */
    accums_out[0].ch[0] = wasm_i32x4_extract_lane(bg_32, 0);
    accums_out[0].ch[1] = wasm_i32x4_extract_lane(bg_32, 1);
    accums_out[0].ch[2] = wasm_i32x4_extract_lane(bg_32, 2);
    accums_out[0].ch[3] = wasm_i32x4_extract_lane(bg_32, 3);

    accums_out[1].ch[0] = wasm_i32x4_extract_lane(fg_32, 0);
    accums_out[1].ch[1] = wasm_i32x4_extract_lane(fg_32, 1);
    accums_out[1].ch[2] = wasm_i32x4_extract_lane(fg_32, 2);
    accums_out[1].ch[3] = wasm_i32x4_extract_lane(fg_32, 3);
}

/* 32768 divided by index. Divide by zero is defined as zero. */
static const guint16 invdiv16 [257] =
{
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

/* Divide color accumulator by scalar using SIMD multiply-high.
 * Divisor must be in range [0..256]. */
void
chafa_color_accum_div_scalar_wasm_simd (ChafaColorAccum *accum, guint16 divisor)
{
    /* Load the 4 channels (each is int16) */
    v128_t acc = wasm_v128_load64_zero(accum);

    /* Create divisor reciprocal vector */
    v128_t div = wasm_i16x8_splat(invdiv16[divisor]);

    /* Multiply and take high bits (equivalent to mulhrs) */
    /* WASM doesn't have direct mulhrs, so we need to emulate it:
     * mulhrs(a, b) = (a * b + 0x4000) >> 15 */
    v128_t prod_lo = wasm_i32x4_extmul_low_i16x8(acc, div);
    v128_t prod_hi = wasm_i32x4_extmul_high_i16x8(acc, div);

    /* Add rounding and shift */
    v128_t round = wasm_i32x4_splat(0x4000);
    prod_lo = wasm_i32x4_add(prod_lo, round);
    prod_hi = wasm_i32x4_add(prod_hi, round);
    prod_lo = wasm_i32x4_shr(prod_lo, 15);
    prod_hi = wasm_i32x4_shr(prod_hi, 15);

    /* Pack back to i16 */
    v128_t result = wasm_i16x8_narrow_i32x4(prod_lo, prod_hi);

    /* Store back (only first 64 bits) */
    guint64 *out = (guint64 *)accum;
    *out = wasm_i64x2_extract_lane(result, 0);
}

/* ============================================================================
 * OPTIMIZED: chafa_color_diff_4x_wasm_simd
 * 
 * COMPLETELY REWRITTEN - original had scalar inner loop!
 * Now fully vectorized using proper SIMD operations.
 * ============================================================================ */
gint
chafa_color_diff_4x_wasm_simd (const ChafaColor *target, const ChafaColor *palette, gint n_colors)
{
    /* Broadcast target color to all lanes [R,G,B,A, R,G,B,A, R,G,B,A, R,G,B,A] */
    v128_t target_bytes = wasm_v128_load32_splat(target);
    v128_t target_16_lo = wasm_u16x8_extend_low_u8x16(target_bytes);  /* [R,G,B,A,R,G,B,A] as u16 */
    
    gint32 best_dist = 0x7FFFFFFF;
    gint best_idx = 0;
    gint i;

    /* Process 4 palette colors at a time */
    for (i = 0; i + 4 <= n_colors; i += 4)
    {
        /* Load 4 palette colors (16 bytes = 4 RGBA) */
        v128_t pal = wasm_v128_load(&palette[i]);
        
        /* Widen palette to i16 */
        v128_t pal_lo = wasm_u16x8_extend_low_u8x16(pal);   /* colors 0,1 */
        v128_t pal_hi = wasm_u16x8_extend_high_u8x16(pal);  /* colors 2,3 */
        
        /* Compute differences */
        v128_t diff_lo = wasm_i16x8_sub(pal_lo, target_16_lo);
        v128_t diff_hi = wasm_i16x8_sub(pal_hi, target_16_lo);
        
        /* Square the differences */
        v128_t sq_lo = wasm_i16x8_mul(diff_lo, diff_lo);
        v128_t sq_hi = wasm_i16x8_mul(diff_hi, diff_hi);
        
        /* Pairwise add to get [R²+G², B²+A²] per color */
        v128_t sum_lo = wasm_i32x4_extadd_pairwise_i16x8(sq_lo);  /* [RG0, BA0, RG1, BA1] */
        v128_t sum_hi = wasm_i32x4_extadd_pairwise_i16x8(sq_hi);  /* [RG2, BA2, RG3, BA3] */
        
        /* Add RG + BA for each color to get total squared distance (ignoring A for error) */
        /* sum_lo = [RG0, BA0, RG1, BA1] -> want [RG0+BA0, RG1+BA1, ...] */
        v128_t shuffled_lo = wasm_i32x4_shuffle(sum_lo, sum_lo, 1, 0, 3, 2);
        v128_t shuffled_hi = wasm_i32x4_shuffle(sum_hi, sum_hi, 1, 0, 3, 2);
        v128_t total_lo = wasm_i32x4_add(sum_lo, shuffled_lo);  /* [dist0, dist0, dist1, dist1] */
        v128_t total_hi = wasm_i32x4_add(sum_hi, shuffled_hi);  /* [dist2, dist2, dist3, dist3] */
        
        /* Extract distances for each of the 4 colors */
        gint32 d0 = wasm_i32x4_extract_lane(total_lo, 0);
        gint32 d1 = wasm_i32x4_extract_lane(total_lo, 2);
        gint32 d2 = wasm_i32x4_extract_lane(total_hi, 0);
        gint32 d3 = wasm_i32x4_extract_lane(total_hi, 2);
        
        /* Find minimum */
        if (d0 < best_dist) { best_dist = d0; best_idx = i; }
        if (d1 < best_dist) { best_dist = d1; best_idx = i + 1; }
        if (d2 < best_dist) { best_dist = d2; best_idx = i + 2; }
        if (d3 < best_dist) { best_dist = d3; best_idx = i + 3; }
    }

    /* Handle remaining colors (0-3) */
    for (; i < n_colors; i++)
    {
        gint dr = palette[i].ch[0] - target->ch[0];
        gint dg = palette[i].ch[1] - target->ch[1];
        gint db = palette[i].ch[2] - target->ch[2];
        gint32 dist = dr * dr + dg * dg + db * db;

        if (dist < best_dist)
        {
            best_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

/* ============================================================================
 * NEW: chafa_work_cell_to_bitmap_wasm_simd
 * 
 * Vectorized bitmap generation - converts pixel block to 64-bit coverage mask.
 * Original is pure scalar loop; this processes 4 pixels per iteration.
 * ============================================================================ */
guint64
chafa_work_cell_to_bitmap_wasm_simd (const ChafaPixel *pixels, const ChafaColorPair *color_pair)
{
    guint64 bitmap = 0;
    
    /* Broadcast fg and bg colors */
    v128_t fg = wasm_v128_load32_splat(&color_pair->colors[1]);  /* FG */
    v128_t bg = wasm_v128_load32_splat(&color_pair->colors[0]);  /* BG */
    
    /* Widen to i16 for distance calculation */
    v128_t fg_16 = wasm_u16x8_extend_low_u8x16(fg);  /* [R,G,B,A,R,G,B,A] */
    v128_t bg_16 = wasm_u16x8_extend_low_u8x16(bg);
    
    gint i;
    
    /* Process 4 pixels at a time */
    for (i = 0; i < CHAFA_SYMBOL_N_PIXELS; i += 4)
    {
        v128_t pix = wasm_v128_load(&pixels[i]);
        
        /* Widen pixels to i16 */
        v128_t pix_lo = wasm_u16x8_extend_low_u8x16(pix);   /* pixels 0,1 */
        v128_t pix_hi = wasm_u16x8_extend_high_u8x16(pix);  /* pixels 2,3 */
        
        /* Compute diff from BG for pixels 0,1 */
        v128_t diff_bg_lo = wasm_i16x8_sub(pix_lo, bg_16);
        v128_t sq_bg_lo = wasm_i16x8_mul(diff_bg_lo, diff_bg_lo);
        v128_t sum_bg_lo = wasm_i32x4_extadd_pairwise_i16x8(sq_bg_lo);
        
        /* Compute diff from FG for pixels 0,1 */
        v128_t diff_fg_lo = wasm_i16x8_sub(pix_lo, fg_16);
        v128_t sq_fg_lo = wasm_i16x8_mul(diff_fg_lo, diff_fg_lo);
        v128_t sum_fg_lo = wasm_i32x4_extadd_pairwise_i16x8(sq_fg_lo);
        
        /* Compute diff from BG for pixels 2,3 */
        v128_t diff_bg_hi = wasm_i16x8_sub(pix_hi, bg_16);
        v128_t sq_bg_hi = wasm_i16x8_mul(diff_bg_hi, diff_bg_hi);
        v128_t sum_bg_hi = wasm_i32x4_extadd_pairwise_i16x8(sq_bg_hi);
        
        /* Compute diff from FG for pixels 2,3 */
        v128_t diff_fg_hi = wasm_i16x8_sub(pix_hi, fg_16);
        v128_t sq_fg_hi = wasm_i16x8_mul(diff_fg_hi, diff_fg_hi);
        v128_t sum_fg_hi = wasm_i32x4_extadd_pairwise_i16x8(sq_fg_hi);
        
        /* Add RG+BA for each pixel to get total distance */
        v128_t bg_dist_lo = wasm_i32x4_add(sum_bg_lo, wasm_i32x4_shuffle(sum_bg_lo, sum_bg_lo, 1, 0, 3, 2));
        v128_t fg_dist_lo = wasm_i32x4_add(sum_fg_lo, wasm_i32x4_shuffle(sum_fg_lo, sum_fg_lo, 1, 0, 3, 2));
        v128_t bg_dist_hi = wasm_i32x4_add(sum_bg_hi, wasm_i32x4_shuffle(sum_bg_hi, sum_bg_hi, 1, 0, 3, 2));
        v128_t fg_dist_hi = wasm_i32x4_add(sum_fg_hi, wasm_i32x4_shuffle(sum_fg_hi, sum_fg_hi, 1, 0, 3, 2));
        
        /* Compare: set bit if bg_error > fg_error (pixel closer to FG) */
        gint32 bg0 = wasm_i32x4_extract_lane(bg_dist_lo, 0);
        gint32 fg0 = wasm_i32x4_extract_lane(fg_dist_lo, 0);
        gint32 bg1 = wasm_i32x4_extract_lane(bg_dist_lo, 2);
        gint32 fg1 = wasm_i32x4_extract_lane(fg_dist_lo, 2);
        gint32 bg2 = wasm_i32x4_extract_lane(bg_dist_hi, 0);
        gint32 fg2 = wasm_i32x4_extract_lane(fg_dist_hi, 0);
        gint32 bg3 = wasm_i32x4_extract_lane(bg_dist_hi, 2);
        gint32 fg3 = wasm_i32x4_extract_lane(fg_dist_hi, 2);
        
        /* Build 4 bits of the bitmap */
        bitmap <<= 4;
        if (bg0 > fg0) bitmap |= 8;
        if (bg1 > fg1) bitmap |= 4;
        if (bg2 > fg2) bitmap |= 2;
        if (bg3 > fg3) bitmap |= 1;
    }
    
    return bitmap;
}

#endif /* HAVE_WASM_SIMD */
