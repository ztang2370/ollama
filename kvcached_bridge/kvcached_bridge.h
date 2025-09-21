#ifndef KVCACHED_BRIDGE_H
#define KVCACHED_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Operation types for the bridge message
typedef enum {
    BRIDGE_OP_INIT = 0,
    BRIDGE_OP_ALLOC_KV_CACHE = 1,
    BRIDGE_OP_ALLOC_KV_BRIDGE = 2,
    BRIDGE_OP_FREE_KV = 3,
    BRIDGE_OP_SHUTDOWN = 4
} bridge_operation_t;

// Logging levels for the bridge
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
} log_level_t;

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

// Set logging level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
void kvcached_bridge_set_log_level(int level);

#ifdef __cplusplus
}
#endif

#endif // KVCACHED_BRIDGE_H