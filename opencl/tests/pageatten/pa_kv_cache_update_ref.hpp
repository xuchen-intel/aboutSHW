/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <cm/cm.h>
#include <cm/cmtl.h>

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

constexpr uint wg_size = WG_SIZE;

#if KV_CACHE_COMPRESSION_PER_TOKEN == 2
constexpr uint MAX_SEQS = 256;
#endif

template <int HEAD_SIZE>
CM_INLINE void load_kvcache(vector_ref<half, HEAD_SIZE> kv_data, const half* kv_ptr [[type("svmptr_t")]], uint offset) {
    if constexpr (HEAD_SIZE == 16 || HEAD_SIZE == 32 || HEAD_SIZE == 64 || HEAD_SIZE == 128) {
        kv_data.format<int>() = cm_ptr_load<int, HEAD_SIZE / 2>((int*)kv_ptr, offset);
    } else if constexpr (HEAD_SIZE == 96) {
        kv_data.select<64, 1>(0).format<int>() = cm_ptr_load<int, 32>((int*)kv_ptr, offset);
        kv_data.select<32, 1>(64).format<int>() = cm_ptr_load<int, 16>((int*)kv_ptr, offset + 32 * sizeof(int));
    } else {
        // # head_size is restricted to by be divisible by 16 in CM PA/Xattn pipeline.
        #pragma unroll
        for(int i = 0; i < HEAD_SIZE / 16; i++) {
            kv_data.select<16, 1>(16*i).format<int>() = cm_ptr_load<int, 8>((int*)kv_ptr, offset + i * 8 * sizeof(int));
        }
    }
}

template <int HEAD_SIZE>
CM_INLINE void load_kvcache(vector_ref<uchar, HEAD_SIZE> kv_data, const uchar* kv_ptr [[type("svmptr_t")]], uint offset) {
    if constexpr (HEAD_SIZE == 16 || HEAD_SIZE == 32 || HEAD_SIZE == 64 || HEAD_SIZE == 128) {
        kv_data.format<int>() = cm_ptr_load<int, HEAD_SIZE / 4>((int*)kv_ptr, offset);
    } else if constexpr (HEAD_SIZE == 96) {
        kv_data.select<64, 1>(0).format<int>() = cm_ptr_load<int, 16>((int*)kv_ptr, offset);
        kv_data.select<32, 1>(64).format<int>() = cm_ptr_load<int, 8>((int*)kv_ptr, offset + 16 * sizeof(int));
    } else {
        #pragma unroll
        for(int i = 0; i < HEAD_SIZE / 16; i++) {
            kv_data.select<16, 1>(16*i).format<int>() = cm_ptr_load<int, 4>((int*)kv_ptr, offset + i * 4 * sizeof(int));
        }
    }
}

template <typename T, int HEAD_SIZE>
CM_INLINE void store_kvcache(const svmptr_t kv_cache [[type("svmptr_t")]], uint offset, vector_ref<T, HEAD_SIZE> kv_data) {
    if constexpr(std::is_same<T, half>::value) {
        if constexpr (HEAD_SIZE == 16 || HEAD_SIZE == 32 || HEAD_SIZE == 64 || HEAD_SIZE == 128) {
            cm_ptr_store<int, HEAD_SIZE / 2>((int*)kv_cache, offset, kv_data.format<int>());
        } else if constexpr (HEAD_SIZE == 96) {
            cm_ptr_store<int, 32>((int*)kv_cache, offset, kv_data.select<64, 1>(0).format<int>());
            cm_ptr_store<int, 16>((int*)kv_cache, offset + 32 * sizeof(int), kv_data.select<32, 1>(64).format<int>());
        } else {
            // # head_size is restricted to by be divisible by 16 in CM PA/Xattn pipeline.
            #pragma unroll
            for(int i = 0; i < HEAD_SIZE / 16; i++) {
                cm_ptr_store<int, 8>((int*)kv_cache, offset + i * 8 * sizeof(int), kv_data.select<16, 1>(16*i).format<int>());
            }
        }
    } else {
        if constexpr (HEAD_SIZE == 32 || HEAD_SIZE == 64 || HEAD_SIZE == 128 || HEAD_SIZE == 256) {
            cm_ptr_store<int, HEAD_SIZE / 4>((int*)kv_cache, offset, kv_data.format<int>());
        } else if constexpr (HEAD_SIZE == 96) {
            cm_ptr_store<int, 16>((int*)kv_cache, offset, kv_data.select<64, 1>(0).format<int>());
            cm_ptr_store<int, 8>((int*)kv_cache, offset + 16 * sizeof(int), kv_data.select<32, 1>(64).format<int>());
        } else {
            // # head_size is restricted to by be divisible by 16 in CM PA/Xattn pipeline.
            #pragma unroll
            for(int i = 0; i < HEAD_SIZE / 16; i++) {
                cm_ptr_store<int, 4>((int*)kv_cache, offset + i * 4 * sizeof(int), kv_data.select<16, 1>(16*i).format<int>());
            }
        }
    }
}

#if KV_CACHE_COMPRESSION_PER_TOKEN == 1
template <int HEAD_SIZE>
CM_INLINE void process_quantization_per_token(const half* in, uchar* out, uint in_offset, uint data_offset, uint scale_offset) {
        vector<half, HEAD_SIZE> in_data;
        load_kvcache<HEAD_SIZE>(in_data, in, in_offset * (int)sizeof(half));

        half max_val = cm_reduced_max<half>(in_data);
        half min_val = cm_reduced_min<half>(in_data);

        half qrange = max_val - min_val;
        float scale_val = qrange == (half)0.0 ? 1.0f : 255.0 / qrange;
        half zp_val = (0.0 - min_val) * scale_val;
        vector<half, HEAD_SIZE>  dequant_data = cm_mul<half>(in_data, scale_val) + zp_val;
        vector<uchar, HEAD_SIZE> data_u8 = cm_rnde<uchar, HEAD_SIZE>(dequant_data);

        store_kvcache<uchar, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + data_offset), 0, data_u8);

        half *out_scale_zp = (half*)(out + scale_offset);
        out_scale_zp[0] = 1.0 / scale_val;
        out_scale_zp[BLOCK_SIZE] = zp_val;
}

#elif KV_CACHE_COMPRESSION_PER_TOKEN == 2
template <int HEAD_SIZE>
CM_INLINE void process_quantization_per_channel(const half* in, uchar* out, uint in_offset, uint data_offset, uint scale_offset, uint pitch, uint dequant_size, uint cur_sub_block_size) {
    matrix<half, SUB_BLOCK_SIZE, HEAD_SIZE> in_data;
    #pragma unroll
    for (int i = 0; i < cur_sub_block_size; i++) {
        load_kvcache<HEAD_SIZE>(in_data.row(i), in, (in_offset + i * pitch) * (int)sizeof(half));
    }
    int offset = 74;
    vector<half, HEAD_SIZE> max_vals = in_data.row(0);
    vector<half, HEAD_SIZE> min_vals = in_data.row(0);
    // {
    //     printf("### in_data:\n");
    //     for (uint c = 0; c < HEAD_SIZE; c++) {
    //         printf("%.15f ", (float)in_data.row(0)[c]);
    //         if (c % 16 == 15) printf("\n");
    //     }
    //     printf("\n");
    // }
    #pragma unroll
    for (int i = 1; i < cur_sub_block_size; i++) {
        max_vals = cm_max<half>(max_vals, in_data.row(i));
        min_vals = cm_min<half>(min_vals, in_data.row(i));
    }

    matrix<half, SUB_BLOCK_SIZE, HEAD_SIZE> update_data;
    if (dequant_size) {
        matrix<uchar, SUB_BLOCK_SIZE, HEAD_SIZE> update_data_u8;
        uint zp_offset = scale_offset + BLOCK_SIZE / SUB_BLOCK_SIZE * HEAD_SIZE * sizeof(half);
        uint update_offset = data_offset - dequant_size * HEAD_SIZE;
        vector<half, HEAD_SIZE> scale_stale;
        vector<half, HEAD_SIZE> zp_stale;
        load_kvcache<HEAD_SIZE>(scale_stale, (half*)out, scale_offset);
        load_kvcache<HEAD_SIZE>(zp_stale, (half*)out, zp_offset);
        #pragma unroll
        for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
            // Use SUB_BLOCK_SIZE in for-loop to avoid cm compiler bug regarding tail loop with data type conversion
            if (i < dequant_size) {
                load_kvcache<HEAD_SIZE>(update_data_u8.row(i), out, update_offset + i * HEAD_SIZE);
                update_data.row(i) = cm_mul<half>(update_data_u8.row(i) - zp_stale, scale_stale);
                // {
                //     printf("### zp_stale:\n");
                //     for (uint c = 0; c < HEAD_SIZE; c++) {
                //         if (c >= offset) continue;
                //         printf("%.15f ", (float)zp_stale[c]);
                //         if (c % 16 == 15) printf("\n");
                //     }
                //     printf("\n");
                // }
                // {
                //     printf("### scale_stale:\n");
                //     for (uint c = 0; c < HEAD_SIZE; c++) {
                //         if (c >= offset) continue;
                //         printf("%.15f ", (float)scale_stale[c]);
                //         if (c % 16 == 15) printf("\n");
                //     }
                //     printf("\n");
                // }
                {
                    printf("### update_data:\n");
                    for (uint c = 0; c < HEAD_SIZE; c++) {
                        if (c >= offset) continue;
                        printf("%.15f ", (float)update_data.row(i)[c]);
                        if (c % 16 == 15) printf("\n");
                    }
                    printf("\n");
                }
                max_vals = cm_max<half>(max_vals, update_data.row(i));
                min_vals = cm_min<half>(min_vals, update_data.row(i));
                // {
                //     printf("### max_vals:\n");
                //     for (uint c = 0; c < HEAD_SIZE; c++) {
                //         printf("%.15f ", (float)max_vals[c]);
                //         if (c % 16 == 15) printf("\n");
                //     }
                //     printf("\n");
                // }
            }
        }
    }

    vector<half, HEAD_SIZE> qrange = max_vals - min_vals;
    vector<ushort, HEAD_SIZE> mask = (qrange == (half)0.0);

    // scale_vals needs fp32 precision to avoid accuracy loss caused by instruction level truncation of fp16 division
    vector<float, HEAD_SIZE> scale_vals = 255.0 / qrange;
    scale_vals.merge(1.0f, mask);
    vector<half, HEAD_SIZE> zp_vals = cm_mul<half>((0.0 - min_vals), scale_vals);

    // half a = 255;
    // half b = 0.04248046875;
    // half cc = a / b;
    // printf("##### a: %.15f\n", (float)a);
    // printf("##### b: %.15f\n", (float)b);
    // printf("##### cc: %.15f\n", (float)cc);

    // half d = 255;
    // half e = 0.0068359375;
    // half f = (float)d / (float)e;
    // printf("##### d: %.15f\n", (float)d);
    // printf("##### e: %.15f\n", (float)e);
    // printf("##### f: %.15f\n", (float)f);

    // half g = 255;
    // float h = qrange[78];
    // if (e == h) {
    //     printf("##### e equals h\n");
    // } else {
    //     printf("##### e not equals h\n");
    // }
    // float j = g / h;
    // half jj = (half)j;
    // printf("##### g: %.15f\n", (float)g);
    // printf("##### h: %.15f\n", (float)h);
    // printf("##### j: %.15f\n", (float)j);
    // printf("##### jj: %.15f\n", (float)jj);

    // half a = 0.8251953125;
    // half b = 6004;
    // half cc = -4700;
    // half d = cm_mul<half>(a, b) + cc;
    // printf("##### a: %.15f\n", (float)a);
    // printf("##### b: %.15f\n", (float)b);
    // printf("##### cc: %.15f\n", (float)cc);
    // printf("##### d: %.15f\n", (float)d);

    // vector<half, 2> e = 0.8251953125;
    // vector<half, 2> f = 6004;
    // vector<half, 2> g = -4700;
    // vector<half, 2> h = cm_mul<half>(e, f) + g;
    // printf("##### e[1]: %.15f\n", (float)e[1]);
    // printf("##### f[1]: %.15f\n", (float)f[1]);
    // printf("##### g[1]: %.15f\n", (float)g[1]);
    // printf("##### h[1]: %.15f\n", (float)h[1]);

    // matrix<half, SUB_BLOCK_SIZE, HEAD_SIZE> l = 0.8251953125;
    // // matrix<half, SUB_BLOCK_SIZE, HEAD_SIZE> l = update_data;
    // if (l.row(0)[73] == update_data.row(0)[73]) {
    //     printf("##### l.row(0)[73] equals update_data.row(0)[73]\n");
    // } else {
    //     printf("##### l.row(0)[73] not equals update_data.row(0)[73]\n");
    //     printf("##### l: %.15f\n", (float)l.row(0)[73]);
    //     printf("##### u: %.15f\n", (float)update_data.row(0)[73]);
    // }
    // vector<half, HEAD_SIZE> m = 6004;
    // vector<half, HEAD_SIZE> n = -4700;
    // vector<half, HEAD_SIZE> o = cm_mul<half>(l.row(0), m) + n;
    // printf("##### l.row(0)[73]: %.15f\n", (float)l.row(0)[73]);
    // printf("##### m[73]: %.15f\n", (float)m[73]);
    // printf("##### n[73]: %.15f\n", (float)n[73]);
    // printf("##### o[73]: %.15f\n", (float)o[73]);

    if (dequant_size) {
        uint update_offset = data_offset - dequant_size * HEAD_SIZE;
        #pragma unroll
        for (int i = 0; i < SUB_BLOCK_SIZE; i++) {
            if (i < dequant_size) {
                vector<half, HEAD_SIZE> quant_data = cm_mul<half>(update_data.row(i), scale_vals) + zp_vals;
                {
                    half a = update_data.row(0)[73];
                    float b = scale_vals[73];
                    half cc = zp_vals[73];
                    half d = a * b;
                    printf("### a: %.15f\n", (float)a);
                    printf("### b: %.15f\n", (float)b);
                    // printf("### cc: %.15f\n", (float)cc);
                    printf("### d: %.15f\n", (float)d);
                }
                // {
                //     printf("### quant_data_before_round:\n");
                //     printf("update_data.row(0)[73]: %.15f\n", (float)update_data.row(0)[73]);
                //     printf("scale_vals[73]: %.15f\n", (float)scale_vals[73]);
                //     printf("zp_vals[73]: %.15f\n", (float)zp_vals[73]);
                //     printf("quant_data[73]: %.15f\n", (float)quant_data[73]);
                //     for (uint c = 0; c < HEAD_SIZE; c++) {
                //         if (c >= offset) continue;
                //         printf("%.15f ", (float)quant_data[c]);
                //         if (c % 16 == 15) printf("\n");
                //     }
                //     printf("\n");
                // }
                quant_data = cm_min<half>(cm_max<half>(quant_data, (half)0.0), (half)255.0);
                vector<uchar, HEAD_SIZE> data_u8 = cm_rnde<uchar, HEAD_SIZE>(quant_data);
                {
                    printf("### qrange:\n");
                    for (uint c = 0; c < HEAD_SIZE; c++) {
                        if (c >= offset) continue;
                        printf("%.15f ", (float)qrange[c]);
                        if (c % 16 == 15) printf("\n");
                    }
                    printf("\n");
                }
                {
                    printf("### scale_vals:\n");
                    for (uint c = 0; c < HEAD_SIZE; c++) {
                        if (c >= offset) continue;
                        printf("%.15f ", (float)scale_vals[c]);
                        if (c % 16 == 15) printf("\n");
                    }
                    printf("\n");
                }
                {
                    printf("### zp_vals:\n");
                    for (uint c = 0; c < HEAD_SIZE; c++) {
                        if (c >= offset) continue;
                        printf("%.15f ", (float)zp_vals[c]);
                        if (c % 16 == 15) printf("\n");
                    }
                    printf("\n");
                }
                // {
                //     printf("### quant_data:\n");
                //     for (uint c = 0; c < HEAD_SIZE; c++) {
                //         if (c >= offset) continue;
                //         printf("%.15f ", (float)quant_data[c]);
                //         if (c % 16 == 15) printf("\n");
                //     }
                //     printf("\n");
                // }
                {
                    printf("### data_u8:\n");
                    for (uint c = 0; c < HEAD_SIZE; c++) {
                        if (c >= offset) continue;
                        printf("%.15f ", (float)data_u8[c]);
                        if (c % 16 == 15) printf("\n");
                    }
                    printf("\n");
                }
                store_kvcache<uchar, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + update_offset + i * HEAD_SIZE), 0, data_u8);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < cur_sub_block_size; i++) {
        vector<half, HEAD_SIZE> quant_data = cm_mul<half>(in_data.row(i), scale_vals) + zp_vals;
        quant_data = cm_min<half>(cm_max<half>(quant_data, (half)0.0), (half)255.0);
        vector<uchar, HEAD_SIZE> data_u8 = cm_rnde<uchar, HEAD_SIZE>(quant_data);
        store_kvcache<uchar, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + data_offset + i * HEAD_SIZE), 0, data_u8);
    }

    vector<half, HEAD_SIZE> scale_out = 1.0 / scale_vals;
    // printf("############ scale_vals[78]: %.15f\n", (float)scale_vals[78]);
    // printf("############ scale_out[78]: %.15f\n", (float)scale_out[78]);
    {
        printf("### scale_out:\n");
        for (uint c = 0; c < HEAD_SIZE; c++) {
            if (c >= offset) continue;
            printf("%.15f ", (float)scale_out[c]);
            if (c % 16 == 15) printf("\n");
        }
        printf("\n");
    }
    vector<half, HEAD_SIZE> zp_out = zp_vals;
    uint zp_offset = scale_offset + BLOCK_SIZE / SUB_BLOCK_SIZE * HEAD_SIZE * sizeof(half);
    store_kvcache<half, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + scale_offset), 0, scale_out);
    store_kvcache<half, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + zp_offset), 0, zp_out);
}

#else
template <int HEAD_SIZE>
CM_INLINE void process_no_quantization(const half* in, uchar* out, uint in_offset, uint out_offset) {
    vector<half, K_HEAD_SIZE> in_data;
    load_kvcache<K_HEAD_SIZE>(in_data, in, in_offset * (int)sizeof(half));
    store_kvcache<half, K_HEAD_SIZE>(reinterpret_cast<svmptr_t>(out), out_offset * (int)sizeof(half), in_data);
}
#endif

template <int HEAD_SIZE, int ADJUSTED_HEAD_SIZE>
CM_INLINE void process_kv_cache_update(
    const half* data,
#if KV_CACHE_COMPRESSION_PER_TOKEN
    uint8_t* cache,
#else
    half* cache,
#endif
    const int32_t* block_indices,
    uint token_idx,
    uint head_idx,
    uint pitch,
    uint offset,
    uint block_offset,
    uint token_start_pos,
    uint dequant_size,
    uint cur_sub_block_size) {
    uint in_offset = token_idx * pitch + head_idx * HEAD_SIZE + offset;
    uint block_k_base_offset = (block_indices[block_offset] * KV_HEADS_NUM + head_idx) * ADJUSTED_BLOCK_SIZE * ADJUSTED_HEAD_SIZE;
    uint data_offset = block_k_base_offset + token_start_pos * HEAD_SIZE;
    uint scale_start_pos = KV_CACHE_COMPRESSION_PER_TOKEN == 2 ? token_start_pos / SUB_BLOCK_SIZE * HEAD_SIZE : token_start_pos;;
    uint scale_offset = block_k_base_offset + HEAD_SIZE * BLOCK_SIZE + scale_start_pos * sizeof(half);

#if KV_CACHE_COMPRESSION_PER_TOKEN == 1
    process_quantization_per_token<HEAD_SIZE>(data, (uchar*)cache, in_offset, data_offset, scale_offset);
#elif KV_CACHE_COMPRESSION_PER_TOKEN == 2
    process_quantization_per_channel<HEAD_SIZE>(data, (uchar*)cache, in_offset, data_offset, scale_offset, pitch, dequant_size, cur_sub_block_size);
#else
    process_no_quantization<HEAD_SIZE>(data, (uchar*)cache, in_offset, data_offset);
#endif
}

extern "C" _GENX_MAIN_ void pa_kv_cache_update(
    const half* key [[type("svmptr_t")]],
    const half* value [[type("svmptr_t")]],
    const int32_t* past_lens [[type("svmptr_t")]],
    const int32_t* block_indices [[type("svmptr_t")]],
    const int32_t* block_indices_begins [[type("svmptr_t")]],
    const int32_t* subsequence_begins [[type("svmptr_t")]],
#if KV_CACHE_COMPRESSION_PER_TOKEN
    uint8_t* key_cache [[type("svmptr_t")]],
    uint8_t* value_cache [[type("svmptr_t")]],
#else
    half* key_cache [[type("svmptr_t")]],
    half* value_cache [[type("svmptr_t")]],
#endif
    uint32_t key_pitch,
    uint32_t key_offset,
    uint32_t value_pitch,
    uint32_t value_offset,
    uint32_t batch_size_in_sequences) {
    // # key:   [batch_size_in_tokens, num_kv_heads * k_head_size]
    // # value  [batch_size_in_tokens, num_kv_heads * v_head_size]
    // # key_cache:   [num_blocks, num_heads, block_size, k_head_size]
    // # value_cache: [num_blocks, num_heads, block_size, v_head_size]
    // 
    // # past_lens: [sequences_num]
    // # subsequence_begins: [sequences_num + 1]
    // # block_indices: [used_blocks_num]
    // # block_indices_begins: [sequences_num + 1]

    // wg_count = aligned_to(batch_size_in_tokens, wg_size) // wg_size
    // # GWS [1, num_heads, wg_count * wg_size]
    // # LWS [1, 1, wg_size]

    static_assert(K_HEAD_SIZE % 16 == 0 && V_HEAD_SIZE % 16 == 0);

    const auto head_idx = cm_group_id(1);
    const auto wg_id = cm_group_id(2);

    if (head_idx != 1) return;

    const uint global_token_idx = KV_CACHE_COMPRESSION_PER_TOKEN == 2 ? cm_global_id(2) * SUB_BLOCK_SIZE : cm_global_id(2);

    uint token_idx = global_token_idx;
    uint dequant_size = 0;
    uint cur_sub_block_size = SUB_BLOCK_SIZE;

#if KV_CACHE_COMPRESSION_PER_TOKEN == 2
    cm_assert(batch_size_in_sequences <= MAX_SEQS);
    uint past_tail_lens[MAX_SEQS];
    uint pad_lens[MAX_SEQS];
    for (uint i = 0; i < batch_size_in_sequences; i++) {
        past_tail_lens[i] = past_lens[i] % SUB_BLOCK_SIZE;
        pad_lens[i] = (SUB_BLOCK_SIZE - (past_tail_lens[i] + subsequence_begins[i + 1] - subsequence_begins[i]) % SUB_BLOCK_SIZE) % SUB_BLOCK_SIZE;
    }
    for (uint i = 0; i < batch_size_in_sequences; i++) {
        if (token_idx >= subsequence_begins[i] + past_tail_lens[i]) {
            token_idx -= past_tail_lens[i];
        }
        if (token_idx >= subsequence_begins[i + 1] + pad_lens[i]) {
            token_idx -= pad_lens[i];
        }
    }
#endif

// token_idx -> subsequence_idx
if (token_idx >= subsequence_begins[batch_size_in_sequences]) return;

#if KV_CACHE_COMPRESSION_PER_TOKEN == 2
    uint finish = 0;
    for (int i = batch_size_in_sequences - 1; i >= 0 ; i--) {
        // last sub-block
        if (token_idx + SUB_BLOCK_SIZE > subsequence_begins[i + 1]) {
            cur_sub_block_size = subsequence_begins[i + 1] - token_idx;
            finish = 1;
        }
        // first sub-block
        if (token_idx == subsequence_begins[i]) {
            dequant_size = past_tail_lens[i];
            finish = 1;
        }
        //middle sub-block
        if (token_idx > subsequence_begins[i]) {
            finish = 1;
        }
        if (finish) {
            break;
        }
    }

    printf("global_token_idx: %d, token_idx: %d, dequant_size: %d, cur_sub_block_size: %d\n", global_token_idx, token_idx, dequant_size, cur_sub_block_size);
#endif

    uint subsequence_idx = 0;
    for (uint i = 0; i < batch_size_in_sequences; i++) {
        if (token_idx >= subsequence_begins[i] && token_idx < subsequence_begins[i + 1]) {
            subsequence_idx = i;
            break;
        }
    }
    // printf("wg:%d.%d, token_idx: %d, subsequence_idx: %d\n", wg_id, wg_local_id, token_idx, subsequence_idx);

    const uint subsequence_begin_idx = subsequence_begins[subsequence_idx];
    const uint past_len = past_lens[subsequence_idx];
    const uint current_block_idx = (past_len + token_idx - subsequence_begin_idx) / BLOCK_SIZE;
    const uint token_start_pos = (past_len + token_idx - subsequence_begin_idx) % BLOCK_SIZE;
    const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;

    process_kv_cache_update<K_HEAD_SIZE, ADJUSTED_K_HEAD_SIZE>(key, key_cache, block_indices, token_idx, head_idx, key_pitch, key_offset, block_offset, token_start_pos, dequant_size, cur_sub_block_size);
    process_kv_cache_update<V_HEAD_SIZE, ADJUSTED_V_HEAD_SIZE>(value, value_cache, block_indices, token_idx, head_idx, value_pitch, value_offset, block_offset, token_start_pos, dequant_size, cur_sub_block_size);
}