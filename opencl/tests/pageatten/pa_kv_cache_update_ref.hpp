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
CM_INLINE void process_quantization_per_channel(const half* in, uchar* out, uint in_offset, uint data_offset, uint scale_offset, uint pitch, uint cur_sub_block_size) {
    matrix<half, SUB_BLOCK_SIZE, HEAD_SIZE> in_data;
    #pragma unroll
    for (int i = 0; i < cur_sub_block_size; i++) {
        load_kvcache<HEAD_SIZE>(in_data.row(i), in, (in_offset + i * pitch) * (int)sizeof(half));
    }

    vector<half, HEAD_SIZE> max_vals = in_data.row(0);
    vector<half, HEAD_SIZE> min_vals = in_data.row(0);
    #pragma unroll
    for (int i = 1; i < cur_sub_block_size; i++) {
        max_vals = cm_max<half>(max_vals, in_data.row(i));
        min_vals = cm_min<half>(min_vals, in_data.row(i));
    }

    vector<half, HEAD_SIZE> qrange = max_vals - min_vals;
    vector<ushort, HEAD_SIZE> mask = (qrange == (half)0.0);

    vector<float, HEAD_SIZE> scale_vals = 255.0 / qrange;
    scale_vals.merge(1.0f, mask);
    vector<half, HEAD_SIZE> zp_vals = cm_mul<half>((0.0 - min_vals), scale_vals);
    #pragma unroll
    for (int i = 0; i < cur_sub_block_size; i++) {
        vector<half, HEAD_SIZE> dequant_data = cm_mul<half>(in_data.row(i), scale_vals) + zp_vals;
        vector<uchar, HEAD_SIZE> data_u8 = cm_rnde<uchar, HEAD_SIZE>(dequant_data);
        store_kvcache<uchar, HEAD_SIZE>(reinterpret_cast<svmptr_t>(out + data_offset + i * HEAD_SIZE), 0, data_u8);
    }

    vector<half, HEAD_SIZE> scale_out = 1.0 / scale_vals;
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
    uint cur_sub_block_size) {
    uint in_offset = token_idx * pitch + head_idx * HEAD_SIZE + offset;
    uint block_k_base_offset = (block_indices[block_offset] * KV_HEADS_NUM + head_idx) * ADJUSTED_BLOCK_SIZE * ADJUSTED_HEAD_SIZE;
    uint data_offset = block_k_base_offset + token_start_pos * HEAD_SIZE;
    uint scale_start_pos = KV_CACHE_COMPRESSION_PER_TOKEN == 2 ? token_start_pos / SUB_BLOCK_SIZE * HEAD_SIZE : token_start_pos;;
    uint scale_offset = block_k_base_offset + HEAD_SIZE * BLOCK_SIZE + scale_start_pos * sizeof(half);

#if KV_CACHE_COMPRESSION_PER_TOKEN == 1
    process_quantization_per_token<HEAD_SIZE>(data, (uchar*)cache, in_offset, data_offset, scale_offset);
#elif KV_CACHE_COMPRESSION_PER_TOKEN == 2
    process_quantization_per_channel<HEAD_SIZE>(data, (uchar*)cache, in_offset, data_offset, scale_offset, pitch, cur_sub_block_size);
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

    const uint global_token_idx = KV_CACHE_COMPRESSION_PER_TOKEN == 2 ? cm_global_id(2) * SUB_BLOCK_SIZE : cm_global_id(2);

    uint token_idx = global_token_idx;
    uint global_subsequence_begins[batch_size_in_sequences];
#if KV_CACHE_COMPRESSION_PER_TOKEN == 2
    global_subsequence_begins[0] = 0;
    for (uint i = 1; i <= batch_size_in_sequences; i++) {
        uint past_tail = past_lens[k] % SUB_BLOCK_SIZE;
        uint cur_tokens = subsequence_begins[i] - subsequence_begins[i - 1];
        global_subsequence_begins[i] = global_subsequence_begins[i - 1] + past_tail + cur_tokens;
    }
    for (uint i = batch_size_in_sequences; i >=0; i--) {
        if (token_idx > global_subsequence_begins[i]) {
            for (int k = 0; k <= i; k++) {
                uint past_tail = past_lens[k] % SUB_BLOCK_SIZE;
                token_idx -= past_tail;
            }
            break;
        }
    }
#endif
    const uint cur_sub_block_size = global_subsequence_begins[batch_size_in_sequences] - token_idx < SUB_BLOCK_SIZE ? global_subsequence_begins[batch_size_in_sequences] - token_idx : SUB_BLOCK_SIZE;

    // token_idx -> subsequence_idx
    if (token_idx >= subsequence_begins[batch_size_in_sequences]) return;
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
    if (past_len % SUB_BLOCK_SIZE && token_idx >= subsequence_begin_idx && token_idx < subsequence_begin_idx + SUB_BLOCK_SIZE) {
    }

    process_kv_cache_update<K_HEAD_SIZE, ADJUSTED_K_HEAD_SIZE>(key, key_cache, block_indices, token_idx, head_idx, key_pitch, key_offset, block_offset, token_start_pos, cur_sub_block_size);
    process_kv_cache_update<V_HEAD_SIZE, ADJUSTED_V_HEAD_SIZE>(value, value_cache, block_indices, token_idx, head_idx, value_pitch, value_offset, block_offset, token_start_pos, cur_sub_block_size);
}