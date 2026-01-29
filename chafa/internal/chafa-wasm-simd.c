/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/* Copyright (C) 2018-2025 Hans Petter Jansson
 * Copyright (C) 2025 Luis Montes - WASM SIMD implementation
 *
 * This file is part of Chafa, a program that shows pictures on text terminals.
 *
 * Chafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Chafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Chafa.  If not, see <http://www.gnu.org/licenses/>. */

#include "config.h"

#ifdef HAVE_WASM_SIMD

#include <wasm_simd128.h>
#include <string.h>
#include "chafa.h"
#include "internal/chafa-private.h"

/* WASM SIMD has 128-bit vectors (v128_t), processing 4 pixels at a time.
 * This is similar to SSE but we use wasm_simd128.h intrinsics. */

/* Calculate cell error between pixels and a color pair using symbol mask.
 * This is the hot path - called for every symbol candidate. */
gint
chafa_calc_cell_error_wasm_simd (const ChafaPixel *pixels, const ChafaColorPair *color_pair,
                                  const guint32 *sym_mask_u32)
{
    v128_t err_accum = wasm_i32x4_splat(0);
    v128_t fg_packed = wasm_i32x4_splat(chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_FG]));
    v128_t bg_packed = wasm_i32x4_splat(chafa_color8_to_u32(color_pair->colors[CHAFA_COLOR_PAIR_BG]));
    gint i;

    /* Process 4 pixels at a time */
    for (i = 0; i < CHAFA_SYMBOL_N_PIXELS; i += 4)
    {
        /* Load 4 pixels (each pixel is 4 bytes RGBA) */
        v128_t pix = wasm_v128_load(&pixels[i]);
        v128_t mask = wasm_v128_load(&sym_mask_u32[i]);

        /* Select fg where mask is set, bg where mask is clear */
        v128_t selected = wasm_v128_bitselect(fg_packed, bg_packed, mask);

        /* Compute per-channel differences
         * We need to expand bytes to 16-bit to avoid overflow during multiply */

        /* Extract low 2 pixels (8 bytes -> 8 i16) */
        v128_t pix_lo = wasm_u16x8_extend_low_u8x16(pix);
        v128_t sel_lo = wasm_u16x8_extend_low_u8x16(selected);
        v128_t diff_lo = wasm_i16x8_sub(sel_lo, pix_lo);
        v128_t sq_lo = wasm_i16x8_mul(diff_lo, diff_lo);

        /* Extract high 2 pixels */
        v128_t pix_hi = wasm_u16x8_extend_high_u8x16(pix);
        v128_t sel_hi = wasm_u16x8_extend_high_u8x16(selected);
        v128_t diff_hi = wasm_i16x8_sub(sel_hi, pix_hi);
        v128_t sq_hi = wasm_i16x8_mul(diff_hi, diff_hi);

        /* Horizontal add pairs: R²+G² and B²+A² for each pixel
         * wasm_i32x4_extadd_pairwise_i16x8 adds adjacent i16 pairs into i32 */
        v128_t sum_lo = wasm_i32x4_extadd_pairwise_i16x8(sq_lo);
        v128_t sum_hi = wasm_i32x4_extadd_pairwise_i16x8(sq_hi);

        /* Now we have [R²+G², B²+A², R²+G², B²+A²] for 2 pixels each
         * Add them together: need R²+G²+B² (ignore alpha) */
        /* Shuffle and add to get per-pixel totals */
        v128_t total_lo = wasm_i32x4_add(sum_lo,
            wasm_i32x4_shuffle(sum_lo, sum_lo, 1, 0, 3, 2));
        v128_t total_hi = wasm_i32x4_add(sum_hi,
            wasm_i32x4_shuffle(sum_hi, sum_hi, 1, 0, 3, 2));

        /* Accumulate */
        err_accum = wasm_i32x4_add(err_accum, total_lo);
        err_accum = wasm_i32x4_add(err_accum, total_hi);
    }

    /* Horizontal sum of the 4 lanes */
    v128_t sum1 = wasm_i32x4_add(err_accum,
        wasm_i32x4_shuffle(err_accum, err_accum, 2, 3, 0, 1));
    v128_t sum2 = wasm_i32x4_add(sum1,
        wasm_i32x4_shuffle(sum1, sum1, 1, 0, 3, 2));

    return wasm_i32x4_extract_lane(sum2, 0);
}

/* Extract mean colors for foreground and background pixels.
 * Accumulates pixel values based on symbol coverage mask. */
void
chafa_extract_cell_mean_colors_wasm_simd (const ChafaPixel *pixels, ChafaColorAccum *accums_out,
                                           const guint32 *sym_mask_u32)
{
    v128_t accum_fg = wasm_i32x4_splat(0);
    v128_t accum_bg = wasm_i32x4_splat(0);
    gint i;

    /* Process 4 pixels at a time */
    for (i = 0; i < CHAFA_SYMBOL_N_PIXELS; i += 4)
    {
        /* Load 4 pixels and mask */
        v128_t pix = wasm_v128_load(&pixels[i]);
        v128_t mask = wasm_v128_load(&sym_mask_u32[i]);

        /* Masked pixels for fg (where mask is 0xFF) and bg (where mask is 0x00) */
        v128_t fg_pix = wasm_v128_and(pix, mask);
        v128_t bg_pix = wasm_v128_andnot(pix, mask);

        /* Widen to 16-bit and accumulate
         * First handle low 2 pixels */
        v128_t fg_lo = wasm_u16x8_extend_low_u8x16(fg_pix);
        v128_t bg_lo = wasm_u16x8_extend_low_u8x16(bg_pix);

        /* Widen further to 32-bit for safe accumulation */
        v128_t fg_lo_lo = wasm_u32x4_extend_low_u16x8(fg_lo);
        v128_t fg_lo_hi = wasm_u32x4_extend_high_u16x8(fg_lo);
        v128_t bg_lo_lo = wasm_u32x4_extend_low_u16x8(bg_lo);
        v128_t bg_lo_hi = wasm_u32x4_extend_high_u16x8(bg_lo);

        accum_fg = wasm_i32x4_add(accum_fg, fg_lo_lo);
        accum_fg = wasm_i32x4_add(accum_fg, fg_lo_hi);
        accum_bg = wasm_i32x4_add(accum_bg, bg_lo_lo);
        accum_bg = wasm_i32x4_add(accum_bg, bg_lo_hi);

        /* High 2 pixels */
        v128_t fg_hi = wasm_u16x8_extend_high_u8x16(fg_pix);
        v128_t bg_hi = wasm_u16x8_extend_high_u8x16(bg_pix);

        v128_t fg_hi_lo = wasm_u32x4_extend_low_u16x8(fg_hi);
        v128_t fg_hi_hi = wasm_u32x4_extend_high_u16x8(fg_hi);
        v128_t bg_hi_lo = wasm_u32x4_extend_low_u16x8(bg_hi);
        v128_t bg_hi_hi = wasm_u32x4_extend_high_u16x8(bg_hi);

        accum_fg = wasm_i32x4_add(accum_fg, fg_hi_lo);
        accum_fg = wasm_i32x4_add(accum_fg, fg_hi_hi);
        accum_bg = wasm_i32x4_add(accum_bg, bg_hi_lo);
        accum_bg = wasm_i32x4_add(accum_bg, bg_hi_hi);
    }

    /* Extract accumulated values [R, G, B, A] */
    accums_out[0].ch[0] = wasm_i32x4_extract_lane(accum_bg, 0);
    accums_out[0].ch[1] = wasm_i32x4_extract_lane(accum_bg, 1);
    accums_out[0].ch[2] = wasm_i32x4_extract_lane(accum_bg, 2);
    accums_out[0].ch[3] = wasm_i32x4_extract_lane(accum_bg, 3);

    accums_out[1].ch[0] = wasm_i32x4_extract_lane(accum_fg, 0);
    accums_out[1].ch[1] = wasm_i32x4_extract_lane(accum_fg, 1);
    accums_out[1].ch[2] = wasm_i32x4_extract_lane(accum_fg, 2);
    accums_out[1].ch[3] = wasm_i32x4_extract_lane(accum_fg, 3);
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

/* Calculate color difference (squared Euclidean distance) for 4 colors at once.
 * Returns the minimum distance index. */
gint
chafa_color_diff_4x_wasm_simd (const ChafaColor *target, const ChafaColor *palette, gint n_colors)
{
    v128_t target_v = wasm_v128_load32_splat(target);
    v128_t min_dist = wasm_i32x4_splat(0x7FFFFFFF);
    v128_t min_idx = wasm_i32x4_splat(0);
    gint i;

    for (i = 0; i + 4 <= n_colors; i += 4)
    {
        /* Load 4 palette colors */
        v128_t pal = wasm_v128_load(&palette[i]);

        /* Compute differences for each color */
        /* This is trickier because we need per-color distances */
        /* For now, process sequentially but keep in registers */
        gint32 dists[4];
        gint j;

        for (j = 0; j < 4; j++)
        {
            gint dr = palette[i + j].ch[0] - target->ch[0];
            gint dg = palette[i + j].ch[1] - target->ch[1];
            gint db = palette[i + j].ch[2] - target->ch[2];
            dists[j] = dr * dr + dg * dg + db * db;
        }

        v128_t dist_v = wasm_v128_load(dists);
        v128_t idx_v = wasm_i32x4_make(i, i + 1, i + 2, i + 3);

        /* Update minimums */
        v128_t mask = wasm_i32x4_lt(dist_v, min_dist);
        min_dist = wasm_v128_bitselect(dist_v, min_dist, mask);
        min_idx = wasm_v128_bitselect(idx_v, min_idx, mask);
    }

    /* Find minimum among the 4 lanes */
    gint32 d0 = wasm_i32x4_extract_lane(min_dist, 0);
    gint32 d1 = wasm_i32x4_extract_lane(min_dist, 1);
    gint32 d2 = wasm_i32x4_extract_lane(min_dist, 2);
    gint32 d3 = wasm_i32x4_extract_lane(min_dist, 3);

    gint32 i0 = wasm_i32x4_extract_lane(min_idx, 0);
    gint32 i1 = wasm_i32x4_extract_lane(min_idx, 1);
    gint32 i2 = wasm_i32x4_extract_lane(min_idx, 2);
    gint32 i3 = wasm_i32x4_extract_lane(min_idx, 3);

    gint best_idx = i0;
    gint32 best_dist = d0;

    if (d1 < best_dist) { best_dist = d1; best_idx = i1; }
    if (d2 < best_dist) { best_dist = d2; best_idx = i2; }
    if (d3 < best_dist) { best_dist = d3; best_idx = i3; }

    /* Handle remaining colors */
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

#endif /* HAVE_WASM_SIMD */
