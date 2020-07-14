#ifndef RAMEN_HPP
#define RAMEN_HPP

#include <immintrin.h>

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace Ramen {
    ////////////////////////
    // SIMD Extension
    ////////////////////////
    struct SimdExt {
        static FORCE_INLINE constexpr __m256d _mm256_set1_pd(double d) { return (__m256d) {d, d, d, d}; }

        static FORCE_INLINE constexpr __m256i _mm256_set1_epi64x(long long q) { return (__m256i) {q, q, q, q}; }

        static FORCE_INLINE constexpr __m256d _mm256_castsi256_pd(__m256i v) { return (__m256d) v; }

        static FORCE_INLINE constexpr __m256i _mm256_castpd_si256(__m256d v) { return (__m256i) v; }
    };

    ////////////////////////
    // Constants
    ////////////////////////
    template<size_t ElementSize>
    struct Constants;

    template<>
    struct Constants<8> {
        static constexpr int mantissa_bits = 52;

        static constexpr auto exponent_mask_i = SimdExt::_mm256_set1_epi64x(0xFF'E0'00'00'00'00'00'00);
        static constexpr auto all_ones_i = SimdExt::_mm256_set1_epi64x(0xFF'FF'FF'FF'FF'FF'FF'FF);

        static constexpr auto exponent_bias_f = SimdExt::_mm256_set1_pd(1023.0);
        static constexpr auto pow2_52_f = SimdExt::_mm256_set1_pd(4503599627370496.0);

        static constexpr auto abs_mask_f = SimdExt::_mm256_castsi256_pd(
            SimdExt::_mm256_set1_epi64x(0x7F'FF'FF'FF'FF'FF'FF'FF));
        static constexpr auto sign_mask_f = SimdExt::_mm256_castsi256_pd(
            SimdExt::_mm256_set1_epi64x(0x80'00'00'00'00'00'00'00));

        static constexpr auto nan_f = SimdExt::_mm256_castsi256_pd(SimdExt::_mm256_set1_epi64x(0x7F'F8'00'00'00'00'00'00));
        static constexpr auto inf_f = SimdExt::_mm256_castsi256_pd(SimdExt::_mm256_set1_epi64x(0x7F'F0'00'00'00'00'00'00));

        static constexpr auto zero_f = SimdExt::_mm256_set1_pd(0.0);
        static constexpr auto one_f = SimdExt::_mm256_set1_pd(1.0);
        static constexpr auto two_f = SimdExt::_mm256_set1_pd(2.0);

        static constexpr auto exp_limit_f = SimdExt::_mm256_set1_pd(708.39);
        static constexpr auto exp_c2_f = SimdExt::_mm256_set1_pd(1. / 2.);
        static constexpr auto exp_c3_f = SimdExt::_mm256_set1_pd(1. / 6.);
        static constexpr auto exp_c4_f = SimdExt::_mm256_set1_pd(1. / 24.);
        static constexpr auto exp_c5_f = SimdExt::_mm256_set1_pd(1. / 120.);
        static constexpr auto exp_c6_f = SimdExt::_mm256_set1_pd(1. / 720.);
        static constexpr auto exp_c7_f = SimdExt::_mm256_set1_pd(1. / 5040.);
        static constexpr auto exp_c8_f = SimdExt::_mm256_set1_pd(1. / 40320.);
        static constexpr auto exp_c9_f = SimdExt::_mm256_set1_pd(1. / 362880.);
        static constexpr auto exp_c10_f = SimdExt::_mm256_set1_pd(1. / 3628800.);
        static constexpr auto exp_c11_f = SimdExt::_mm256_set1_pd(1. / 39916800.);
        static constexpr auto exp_c12_f = SimdExt::_mm256_set1_pd(1. / 479001600.);
        static constexpr auto exp_c13_f = SimdExt::_mm256_set1_pd(1. / 6227020800.);

        static constexpr auto tanh_limit_f = SimdExt::_mm256_set1_pd(350.);

        static constexpr auto log2_e_f = SimdExt::_mm256_set1_pd(1.44269504088896340736);
        static constexpr auto ln_2_hi_f = SimdExt::_mm256_set1_pd(0.693145751953125);
        static constexpr auto ln_2_lo_f = SimdExt::_mm256_set1_pd(1.42860682030941723212e-6);
    };

    ////////////////////////
    // Scalar Types
    ////////////////////////
    template<size_t ElementSize>
    struct ScalarTypes;

    template<>
    struct ScalarTypes<8> {
        using float_t = double;
        using int_t = long long;
    };

    ////////////////////////
    // Vector Types
    ////////////////////////
    template<size_t ElementSize, size_t NElements>
    struct VectorTypes;

    template<>
    struct VectorTypes<8, 4> {
        using float_t = __m256d;
        using int_t = __m256i;

        static FORCE_INLINE float_t from(__m256d v) { return v; }
        static FORCE_INLINE int_t from(__m256i v) { return v; }
    };

    template<>
    struct VectorTypes<8, 2> {
        using float_t = __m128d;
        using int_t = __m128i;

        static FORCE_INLINE float_t from(__m256d v) { return _mm256_castpd256_pd128(v); }
        static FORCE_INLINE int_t from(__m256i v) { return _mm256_castsi256_si128(v); }
    };

    template<>
    struct VectorTypes<8, 1> {
        using float_t = __m128d;
        using int_t = __m128i;

        static FORCE_INLINE float_t from(__m256d v) { return _mm256_castpd256_pd128(v); }
        static FORCE_INLINE int_t from(__m256i v) { return _mm256_castsi256_si128(v); }
    };

    ////////////////////////
    // SIMD Types
    ////////////////////////
    template<size_t ElementSize, size_t NElements>
    struct SimdTypes {
        static const size_t element_size = ElementSize;
        static const size_t n_elements = NElements;

        using vt = VectorTypes<ElementSize, NElements>;
        using st = ScalarTypes<ElementSize>;
    };

    ////////////////////////
    // SIMD Operators
    ////////////////////////
    template<size_t ElementSize, size_t NElements>
    struct SimdOps;

    template<>
    struct SimdOps<8, 4> {
        using t = SimdTypes<8, 4>;
        using c = Constants<8>;

        using vt = t::vt;
        using st = t::st;

        // set
        static constexpr auto set1_f = _mm256_set1_pd;
        static constexpr auto set_f = _mm256_set_pd;
        static constexpr auto setr_f = _mm256_set_pd;

        // load
        static constexpr auto load_f = _mm256_load_pd;
        static constexpr auto loadu_f = _mm256_loadu_pd;

        // store
        static constexpr auto store_f = _mm256_store_pd;
        static constexpr auto storeu_f = _mm256_storeu_pd;

        // arithmetic
        static constexpr auto add_f = _mm256_add_pd;
        static constexpr auto sub_f = _mm256_sub_pd;
        static constexpr auto mul_f = _mm256_mul_pd;
        static constexpr auto div_f = _mm256_div_pd;

        // logical
        static constexpr auto and_f = _mm256_and_pd;
        static constexpr auto andnot_f = _mm256_andnot_pd;
        static constexpr auto or_f = _mm256_or_pd;
        static constexpr auto xor_f = _mm256_xor_pd;
        static constexpr auto and_i = _mm256_and_si256;
        static constexpr auto andnot_i = _mm256_andnot_si256;
        static constexpr auto or_i = _mm256_or_si256;
        static constexpr auto xor_i = _mm256_xor_si256;

        // fma
        static constexpr auto fmadd_f = _mm256_fmadd_pd;
        static constexpr auto fmsub_f = _mm256_fmsub_pd;
        static constexpr auto fnmadd_f = _mm256_fnmadd_pd;
        static constexpr auto fnmsub_f = _mm256_fnmsub_pd;

        // math
        static constexpr auto max_f = _mm256_max_pd;
        static constexpr auto min_f = _mm256_min_pd;
        static constexpr auto round_f = [](auto v) { return _mm256_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT); };
        static constexpr auto ceil_f = [](auto v) { return _mm256_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_POS_INF); };
        static constexpr auto floor_f = [](auto v) { return _mm256_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEG_INF); };
        static constexpr auto trunc_f = [](auto v) { return _mm256_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO); };

        // compare
        static constexpr auto eq_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); };
        static constexpr auto ne_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ); };
        static constexpr auto lt_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); };
        static constexpr auto le_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); };
        static constexpr auto gt_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); };
        static constexpr auto ge_f = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); };
        static constexpr auto eq_i = _mm256_cmpeq_epi64;
        static constexpr auto ne_i = [](auto a, auto b) { return xor_i(eq_i(a, b), vt::from(c::all_ones_i)); };

        // swizzle
        static constexpr auto blendv_f = _mm256_blendv_pd;
        static constexpr auto if_f = [](auto m, auto a, auto b) { return blendv_f(b, a, m); };

        // cast
        static constexpr auto cast_f = _mm256_castsi256_pd;
        static constexpr auto cast_i = _mm256_castpd_si256;

        // shift
        static constexpr auto slli_i = _mm256_slli_epi64;

        // extension
        static constexpr auto abs_f = [](auto v) { return and_f(v, vt::from(c::abs_mask_f)); };
        static constexpr auto neg_f = [](auto v) { return xor_f(v, vt::from(c::sign_mask_f)); };
        static constexpr auto sign_f = [](auto v) { return cast_f(ne_i(and_i(cast_i(v), cast_i(vt::from(c::sign_mask_f))), vt::from(c::zero_f))); };
        static constexpr auto is_finite_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(ne_i(exponent, mask));
        };
        static constexpr auto is_inf_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            return cast_f(eq_i(x, mask));
        };
        static constexpr auto is_nan_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(andnot_i(eq_i(x, mask), eq_i(exponent, mask)));
        };
        static constexpr auto hor_f = [](auto v) { return !_mm256_testz_si256(cast_i(v), cast_i(v)); };
        static constexpr auto hand_f = [](auto v) { return _mm256_testc_si256(cast_i(v), vt::from(c::all_ones_i)); };
    };


    template<>
    struct SimdOps<8, 2> {
        using t = SimdTypes<8, 2>;
        using c = Constants<8>;

        using vt = t::vt;
        using st = t::st;

        // set
        static constexpr auto set1_f = _mm_set1_pd;
        static constexpr auto set_f = _mm_set_pd;
        static constexpr auto setr_f = _mm_setr_pd;

        // load
        static constexpr auto load_f = _mm_load_pd;
        static constexpr auto loadu_f = _mm_loadu_pd;

        // store
        static constexpr auto store_f = _mm_store_pd;
        static constexpr auto storeu_f = _mm_storeu_pd;

        // arithmetic
        static constexpr auto add_f = _mm_add_pd;
        static constexpr auto sub_f = _mm_sub_pd;
        static constexpr auto mul_f = _mm_mul_pd;
        static constexpr auto div_f = _mm_div_pd;

        // logical
        static constexpr auto and_f = _mm_and_pd;
        static constexpr auto andnot_f = _mm_andnot_pd;
        static constexpr auto or_f = _mm_or_pd;
        static constexpr auto xor_f = _mm_xor_pd;
        static constexpr auto and_i = _mm_and_si128;
        static constexpr auto andnot_i = _mm_andnot_si128;
        static constexpr auto or_i = _mm_or_si128;
        static constexpr auto xor_i = _mm_xor_si128;

        // fma
        static constexpr auto fmadd_f = _mm_fmadd_pd;
        static constexpr auto fmsub_f = _mm_fmsub_pd;
        static constexpr auto fnmadd_f = _mm_fnmadd_pd;
        static constexpr auto fnmsub_f = _mm_fnmsub_pd;

        // math
        static constexpr auto max_f = _mm_max_pd;
        static constexpr auto min_f = _mm_min_pd;
        static constexpr auto round_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT); };
        static constexpr auto ceil_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_POS_INF); };
        static constexpr auto floor_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEG_INF); };
        static constexpr auto trunc_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO); };

        // compare
        static constexpr auto eq_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_EQ_OQ); };
        static constexpr auto ne_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_NEQ_OQ); };
        static constexpr auto lt_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_LT_OQ); };
        static constexpr auto le_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_LE_OQ); };
        static constexpr auto gt_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_GT_OQ); };
        static constexpr auto ge_f = [](auto a, auto b) { return _mm_cmp_pd(a, b, _CMP_GE_OQ); };
        static constexpr auto eq_i = _mm_cmpeq_epi64;
        static constexpr auto ne_i = [](auto a, auto b) { return andnot_i(eq_i(a, b), vt::from(c::all_ones_i)); };

        // swizzle
        static constexpr auto blendv_f = _mm_blendv_pd;
        static constexpr auto if_f = [](auto m, auto a, auto b) { return blendv_f(b, a, m); };

        // cast
        static constexpr auto cast_f = _mm_castsi128_pd;
        static constexpr auto cast_i = _mm_castpd_si128;

        // shift
        static constexpr auto slli_i = _mm_slli_epi64;

        // extension
        static constexpr auto abs_f = [](auto v) { return and_f(v, vt::from(c::abs_mask_f)); };
        static constexpr auto neg_f = [](auto v) { return xor_f(v, vt::from(c::sign_mask_f)); };
        static constexpr auto sign_f = [](auto v) {
            return cast_f(
                ne_i(and_i(cast_i(v), cast_i(vt::from(c::sign_mask_f))), vt::from(c::zero_f)));
        };
        static constexpr auto is_finite_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(ne_i(exponent, mask));
        };
        static constexpr auto is_inf_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            return cast_f(eq_i(x, mask));
        };
        static constexpr auto is_nan_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(andnot_i(eq_i(x, mask), eq_i(exponent, mask)));
        };
        static constexpr auto hor_f = [](auto v) { return !_mm_testz_si128(cast_i(v), cast_i(v)); };
        static constexpr auto hand_f = [](auto v) { return _mm_testc_si128(cast_i(v), vt::from(c::all_ones_i)); };
    };

    template<>
    struct SimdOps<8, 1> {
        using t = SimdTypes<8, 1>;
        using c = Constants<8>;

        using vt = t::vt;
        using st = t::st;

        // set
        static constexpr auto set1_f = _mm_set_sd;
        static constexpr auto set_f = _mm_set_sd;
        static constexpr auto setr_f = _mm_set_sd;

        // load
        static constexpr auto load_f = _mm_load_sd;
        static constexpr auto loadu_f = _mm_load_sd;

        // store
        static constexpr auto store_f = _mm_store_sd;
        static constexpr auto storeu_f = _mm_store_sd;

        // arithmetic
        static constexpr auto add_f = _mm_add_sd;
        static constexpr auto sub_f = _mm_sub_sd;
        static constexpr auto mul_f = _mm_mul_sd;
        static constexpr auto div_f = _mm_div_sd;

        // logical
        static constexpr auto and_f = _mm_and_pd;
        static constexpr auto andnot_f = _mm_andnot_pd;
        static constexpr auto or_f = _mm_or_pd;
        static constexpr auto xor_f = _mm_xor_pd;
        static constexpr auto and_i = _mm_and_si128;
        static constexpr auto andnot_i = _mm_andnot_si128;
        static constexpr auto or_i = _mm_or_si128;
        static constexpr auto xor_i = _mm_xor_si128;

        // fma
        static constexpr auto fmadd_f = _mm_fmadd_sd;
        static constexpr auto fmsub_f = _mm_fmsub_sd;
        static constexpr auto fnmadd_f = _mm_fnmadd_sd;
        static constexpr auto fnmsub_f = _mm_fnmsub_sd;

        // math
        static constexpr auto max_f = _mm_max_sd;
        static constexpr auto min_f = _mm_min_sd;
        static constexpr auto round_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT); };
        static constexpr auto ceil_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_POS_INF); };
        static constexpr auto floor_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEG_INF); };
        static constexpr auto trunc_f = [](auto v) { return _mm_round_pd(v, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO); };

        // compare
        static constexpr auto eq_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_EQ_OQ); };
        static constexpr auto ne_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_NEQ_OQ); };
        static constexpr auto lt_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_LT_OQ); };
        static constexpr auto le_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_LE_OQ); };
        static constexpr auto gt_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_GT_OQ); };
        static constexpr auto ge_f = [](auto a, auto b) { return _mm_cmp_sd(a, b, _CMP_GE_OQ); };
        static constexpr auto eq_i = _mm_cmpeq_epi64;
        static constexpr auto ne_i = [](auto a, auto b) { return andnot_i(eq_i(a, b), vt::from(c::all_ones_i)); };

        // swizzle
        static constexpr auto blendv_f = _mm_blendv_pd;
        static constexpr auto if_f = [](auto m, auto a, auto b) { return blendv_f(b, a, m); };

        // cast
        static constexpr auto cast_f = _mm_castsi128_pd;
        static constexpr auto cast_i = _mm_castpd_si128;

        // shift
        static constexpr auto slli_i = _mm_slli_epi64;

        // extension
        static constexpr auto abs_f = [](auto v) { return and_f(v, vt::from(c::abs_mask_f)); };
        static constexpr auto neg_f = [](auto v) { return xor_f(v, vt::from(c::sign_mask_f)); };
        static constexpr auto sign_f = [](auto v) {
            return cast_f(
                ne_i(and_i(cast_i(v), cast_i(vt::from(c::sign_mask_f))), vt::from(c::zero_f)));
        };

        static constexpr auto is_finite_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(ne_i(exponent, mask));
        };

        static constexpr auto is_inf_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            return cast_f(eq_i(x, mask));
        };

        static constexpr auto is_nan_f = [](auto v) {
            auto x = slli_i(cast_i(v), 1);
            auto mask = vt::from(c::exponent_mask_i);
            auto exponent = and_i(x, mask);
            return cast_f(andnot_i(eq_i(x, mask), eq_i(exponent, mask)));
        };

        static constexpr auto hor_f = [](auto v) { return _mm_cvtsi128_si64(cast_i(v)) != 0; };
        static constexpr auto hand_f = [](auto v) { return _mm_cvtsi128_si64(cast_i(v)) != 0; };
    };

    ////////////////////////
    // Functions
    ////////////////////////
    template<template<size_t ElementSize, size_t NElements> class Func, size_t ElementSize, size_t NElements>
    struct Processor {
        using ops = SimdOps<ElementSize, NElements>;

        using c = typename ops::c;
        using vt = typename ops::t::vt;
        using st = typename ops::t::st;

        static FORCE_INLINE void call(typename st::float_t *dst, const typename st::float_t *src, int n) {
            if (n >= NElements) {
                ops::storeu_f(dst, Func<ElementSize, NElements>::calc(ops::loadu_f(src)));
                call(dst + NElements, src + NElements, n - NElements);
            } else if constexpr (NElements > 1) {
                Func<ElementSize, (NElements >> 1)>::call(dst, src, n);
            }
        }
    };

    template<size_t ElementSize, size_t NElements>
    struct Pow2 : public Processor<Pow2, ElementSize, NElements> {
        using ops = SimdOps<ElementSize, NElements>;

        using c = typename ops::c;
        using vt = typename ops::t::vt;
        using st = typename ops::t::st;

        static FORCE_INLINE auto calc(const typename vt::float_t v) {
            auto a = ops::add_f(v, ops::add_f(vt::from(c::exponent_bias_f), vt::from(c::pow2_52_f)));
            auto b = ops::cast_i(a);
            auto c = ops::slli_i(b, c::mantissa_bits);

            return ops::cast_f(c);
        }
    };

    template<size_t ElementSize, size_t NElements>
    struct Exp : public Processor<Exp, ElementSize, NElements> {
        using ops = SimdOps<ElementSize, NElements>;

        using c = typename ops::c;
        using vt = typename ops::t::vt;
        using st = typename ops::t::st;

        static FORCE_INLINE auto calc(const typename vt::float_t &v) {
            using pow2 = Pow2<ElementSize, NElements>;

            auto x = v;
            auto n = ops::round_f(ops::mul_f(x, vt::from(c::log2_e_f)));
                 x = ops::fnmadd_f(n, vt::from(c::ln_2_hi_f), x);
                 x = ops::fnmadd_f(n, vt::from(c::ln_2_lo_f), x);
            auto x2 = ops::mul_f(x, x);
            auto x4 = ops::mul_f(x2, x2);
            auto x8 = ops::mul_f(x4, x4);
            auto c = pow2::calc(n);

            auto e = ops::fmadd_f(
                ops::fmadd_f(
                    ops::fmadd_f(vt::from(c::exp_c13_f), x, vt::from(c::exp_c12_f)),
                    x4,
                    ops::fmadd_f(
                        ops::fmadd_f(vt::from(c::exp_c11_f), x, vt::from(c::exp_c10_f)),
                        x2,
                        ops::fmadd_f(vt::from(c::exp_c9_f), x, vt::from(c::exp_c8_f))
                    )
                ),
                x8,
                ops::fmadd_f(
                    ops::fmadd_f(
                        ops::fmadd_f(vt::from(c::exp_c7_f), x, vt::from(c::exp_c6_f)),
                        x2,
                        ops::fmadd_f(vt::from(c::exp_c5_f), x, vt::from(c::exp_c4_f))
                    ),
                    x4,
                    ops::fmadd_f(
                        ops::fmadd_f(vt::from(c::exp_c3_f), x, vt::from(c::exp_c2_f)),
                        x2,
                        x
                    )
                )
            );

            auto valid_mask = ops::and_f(ops::lt_f(ops::abs_f(v), vt::from(c::exp_limit_f)), ops::is_finite_f(v));
            auto e_x = ops::fmadd_f(c, e, c);

            if (ops::hand_f(valid_mask)) {
                return e_x;
            } else {
                auto sign_mask = ops::sign_f(v);
                auto limit = ops::if_f(sign_mask, vt::from(c::zero_f), vt::from(c::inf_f));
                auto nan_mask = ops::is_nan_f(v);
                e_x = ops::if_f(valid_mask, e_x, limit);
                return ops::if_f(nan_mask, e_x, v);
            }
        }
    };

    template<size_t ElementSize, size_t NElements>
    struct Sigmoid : public Processor<Sigmoid, ElementSize, NElements> {
        using ops = SimdOps<ElementSize, NElements>;

        using c = typename ops::c;
        using vt = typename ops::t::vt;
        using st = typename ops::t::st;

        static FORCE_INLINE auto calc(const typename vt::float_t &v) {
            using exp = Exp<ElementSize, NElements>;

            auto inv_e_x = exp::calc(ops::neg_f(v));
            return ops::div_f(vt::from(c::one_f), ops::add_f(vt::from(c::one_f), inv_e_x));
        }
    };

    template<size_t ElementSize, size_t NElements>
    struct Tanh : public Processor<Tanh, ElementSize, NElements> {
        using ops = SimdOps<ElementSize, NElements>;

        using c = typename ops::c;
        using vt = typename ops::t::vt;
        using st = typename ops::t::st;

        static FORCE_INLINE auto calc(const typename vt::float_t &v) {
            using exp = Exp<ElementSize, NElements>;

            auto x = ops::abs_f(v);
            auto e_2x = exp::calc(ops::add_f(x, x));
            auto y = ops::sub_f(
                vt::from(c::one_f),
                ops::div_f(
                    vt::from(c::two_f),
                    ops::add_f(e_2x, vt::from(c::one_f))
                )
            );

            auto limit_mask = ops::gt_f(v, vt::from(c::tanh_limit_f));
                 y = ops::if_f(limit_mask, vt::from(c::one_f), y);

            return ops::xor_f(y, ops::and_f(v, vt::from(c::sign_mask_f)));
        }
    };
}

#endif
