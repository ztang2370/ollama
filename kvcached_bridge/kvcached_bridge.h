#ifndef KVCACHED_BRIDGE_H
#define KVCACHED_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Python bridge
int kvcached_bridge_init();

// Call Python init_kvcached function (Stage 1)
int kvcached_bridge_init_kvcached(const char* device, int async_sched);

// Call Python alloc_kv_cache function (Stage 2)
int kvcached_bridge_alloc_kv_cache(int num_blocks, int block_size, int head_num, int head_dim, int num_layers, const char* device);

// Call Python alloc_kv_bridge function (Stage 3 - allocate blocks for a request)
long long* kvcached_bridge_alloc_kv(int num_blocks);

// Call Python free_kv function (free blocks for a request)
int kvcached_bridge_free_kv(long long* block_ids, int num_blocks);

// Call Python shutdown_kvcached function
int kvcached_bridge_shutdown_kvcached();

// Cleanup function
void kvcached_bridge_cleanup();

#ifdef __cplusplus
}
#endif

#endif // KVCACHED_BRIDGE_H