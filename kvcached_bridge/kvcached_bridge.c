#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "kvcached_bridge.h"
#include <stdarg.h>

// Global Python module reference and thread state
static PyObject* kvcached_module = NULL;
static PyThreadState* main_thread_state = NULL;

// Global logging level - can be changed at runtime
static log_level_t current_log_level = LOG_INFO;

// Simple logging utility function
void cbridge_log(log_level_t level, const char* fmt, ...) {
    if (level < current_log_level) {
        return;  // Skip messages below current log level
    }

    // Level prefix
    const char* level_str;
    switch (level) {
        case LOG_DEBUG: level_str = "DEBUG"; break;
        case LOG_INFO:  level_str = "INFO";  break;
        case LOG_WARN:  level_str = "WARN";  break;
        case LOG_ERROR: level_str = "ERROR"; break;
        default:        level_str = "UNKNOWN"; break;
    }

    fprintf(stderr, "C_BRIDGE [%s]: ", level_str);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
}

// Synchronous bridge using dedicated thread
typedef struct {
    bridge_operation_t type;  // Operation type enum
    union {
        struct {
            const char* device;
            int async_sched;
        } init;
        struct {
            int num_blocks;
            int block_size;
            int head_num;
            int head_dim;
            int num_layers;
            const char* device;
        } alloc_cache;
        struct {
            int num_blocks;
        } alloc_bridge;
        struct {
            long long* block_ids;
            int num_blocks;
        } free;
    } data;
    int result;
    long long* result_blocks;
    int processed;  // Flag to indicate processing is complete
} bridge_message_t;

static pthread_t python_thread;
static int thread_running = 0;
static int thread_initialized = 0;
static pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t queue_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t init_cond = PTHREAD_COND_INITIALIZER;
static bridge_message_t* current_message = NULL;
static int shutdown_requested = 0;

// Python thread function that handles all Python operations
void* python_thread_func(void* arg) {
    // Python is already initialized in main thread
    // Signal that thread is initialized
    pthread_mutex_lock(&queue_mutex);
    thread_initialized = 1;
    pthread_cond_signal(&init_cond);
    pthread_mutex_unlock(&queue_mutex);

    cbridge_log(LOG_DEBUG, "Python thread started");

    while (!shutdown_requested) {
        // Wait for a message
        pthread_mutex_lock(&queue_mutex);
        while (current_message == NULL && !shutdown_requested) {
            pthread_cond_wait(&queue_cond, &queue_mutex);
        }

        if (shutdown_requested) {
            pthread_mutex_unlock(&queue_mutex);
            break;
        }

        bridge_message_t* msg = current_message;
        current_message = NULL;
        pthread_mutex_unlock(&queue_mutex);

        // Process the message
        switch (msg->type) {
            case BRIDGE_OP_INIT: { // init_kvcached
                cbridge_log(LOG_DEBUG, "processing init message");
                PyEval_RestoreThread(main_thread_state);

                // kvcached_module is already imported in main thread

                if (kvcached_module) {
                    // Call init_kvcached
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "init_kvcached");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject* pArgs = PyTuple_New(5);
                        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(0));
                        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(1));
                        PyTuple_SetItem(pArgs, 2, PyBool_FromLong(0));
                        PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(msg->data.init.device));
                        PyTuple_SetItem(pArgs, 4, PyBool_FromLong(msg->data.init.async_sched));

                        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                        if (pResult) {
                            cbridge_log(LOG_DEBUG, "init_kvcached succeeded");
                            msg->result = 0;
                            Py_DECREF(pResult);
                        } else {
                            cbridge_log(LOG_ERROR, "init_kvcached failed");
                            msg->result = -1;
                            PyErr_Print();
                        }
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        cbridge_log(LOG_ERROR, "cannot find init_kvcached function");
                        msg->result = -1;
                    }
                } else {
                    cbridge_log(LOG_ERROR, "failed to import kvcached module");
                    msg->result = -1;
                    PyErr_Print();
                }

                main_thread_state = PyEval_SaveThread();
                cbridge_log(LOG_DEBUG, "finished processing init message, result=%d", msg->result);
                msg->processed = 1;
                break;
            }
            case BRIDGE_OP_ALLOC_KV_CACHE: { // alloc_kv_cache
                cbridge_log(LOG_DEBUG, "processing alloc_kv_cache message");
                PyEval_RestoreThread(main_thread_state);

                if (kvcached_module) {
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "alloc_kv_cache");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        // Create kvcache_shape tuple: (2, num_blocks, head_num, head_dim)
                        PyObject* shape_tuple = PyTuple_New(4);
                        PyTuple_SetItem(shape_tuple, 0, PyLong_FromLong(2));
                        PyTuple_SetItem(shape_tuple, 1, PyLong_FromLong(msg->data.alloc_cache.num_blocks));
                        PyTuple_SetItem(shape_tuple, 2, PyLong_FromLong(msg->data.alloc_cache.head_num));
                        PyTuple_SetItem(shape_tuple, 3, PyLong_FromLong(msg->data.alloc_cache.head_dim));

                        // Create arguments
                        PyObject* pArgs = PyTuple_New(6);
                        PyTuple_SetItem(pArgs, 0, shape_tuple);
                        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(msg->data.alloc_cache.block_size));
                        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString("float16"));
                        PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(msg->data.alloc_cache.device));
                        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(msg->data.alloc_cache.num_layers));
                        PyTuple_SetItem(pArgs, 5, PyUnicode_FromString("MHA"));

                        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                        if (pResult && PyList_Check(pResult)) {
                            cbridge_log(LOG_DEBUG, "alloc_kv_cache succeeded");
                            msg->result = 0;
                            Py_DECREF(pResult);
                        } else {
                            cbridge_log(LOG_ERROR, "alloc_kv_cache failed");
                            msg->result = -1;
                            PyErr_Print();
                        }
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        cbridge_log(LOG_ERROR, "cannot find alloc_kv_cache function");
                        msg->result = -1;
                    }
                } else {
                    cbridge_log(LOG_ERROR, "kvcached module not available");
                    msg->result = -1;
                }

                main_thread_state = PyEval_SaveThread();
                msg->processed = 1;
                break;
            }
            case BRIDGE_OP_ALLOC_KV_BRIDGE: { // alloc_kv_bridge
                PyEval_RestoreThread(main_thread_state);

                if (kvcached_module) {
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "alloc_kv_bridge");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject* pArgs = PyTuple_New(1);
                        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(msg->data.alloc_bridge.num_blocks));

                        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
                        if (pValue && PyList_Check(pValue)) {
                            Py_ssize_t list_size = PyList_Size(pValue);
                            msg->result_blocks = (long long*)malloc(list_size * sizeof(long long));
                            if (msg->result_blocks) {
                                for (Py_ssize_t i = 0; i < list_size; i++) {
                                    PyObject* item = PyList_GetItem(pValue, i);
                                    if (PyLong_Check(item)) {
                                        msg->result_blocks[i] = PyLong_AsLongLong(item);
                                    }
                                }
                                msg->result = 0;
                            } else {
                                msg->result = -1;
                            }
                        } else {
                            msg->result = -1;
                            if (pValue) PyErr_Print();
                        }

                        Py_XDECREF(pValue);
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        msg->result = -1;
                    }
                } else {
                    msg->result = -1;
                }

                main_thread_state = PyEval_SaveThread();
                msg->processed = 1;
                break;
            }
            case BRIDGE_OP_FREE_KV: { // free_kv
                PyEval_RestoreThread(main_thread_state);

                if (kvcached_module) {
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "free_kv_bridge");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject* pList = PyList_New(msg->data.free.num_blocks);
                        for (int i = 0; i < msg->data.free.num_blocks; i++) {
                            PyList_SetItem(pList, i, PyLong_FromLongLong(msg->data.free.block_ids[i]));
                        }

                        PyObject* pArgs = PyTuple_New(1);
                        PyTuple_SetItem(pArgs, 0, pList);

                        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
                        if (pValue && PyLong_Check(pValue)) {
                            msg->result = (int)PyLong_AsLong(pValue);
                        } else {
                            msg->result = -1;
                            if (pValue) PyErr_Print();
                        }

                        Py_XDECREF(pValue);
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        msg->result = -1;
                    }
                } else {
                    msg->result = -1;
                }

                // Don't free block_ids - it's managed by Go
                main_thread_state = PyEval_SaveThread();
                msg->processed = 1;
                break;
            }
            case BRIDGE_OP_SHUTDOWN: { // shutdown
                PyEval_RestoreThread(main_thread_state);

                if (kvcached_module) {
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "shutdown_kvcached");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject* pResult = PyObject_CallObject(pFunc, NULL);
                        if (pResult) {
                            msg->result = 0;
                            Py_DECREF(pResult);
                        } else {
                            msg->result = -1;
                            PyErr_Print();
                        }
                        Py_DECREF(pFunc);
                    }
                    Py_XDECREF(kvcached_module);
                    kvcached_module = NULL;
                }

                main_thread_state = PyEval_SaveThread();
                msg->processed = 1;
                break;
            }
        }
    }

    // Finalize Python
    PyEval_RestoreThread(main_thread_state);
    if (kvcached_module) {
        Py_XDECREF(kvcached_module);
        kvcached_module = NULL;
    }
    Py_Finalize();

    cbridge_log(LOG_DEBUG, "Python thread exiting");
    return NULL;
}

// Initialize the synchronous bridge
int bridge_init() {
    cbridge_log(LOG_DEBUG, "bridge_init called");

    if (thread_running) {
        cbridge_log(LOG_DEBUG, "thread already running");
        return 0;
    }

    // Initialize Python in the main thread
    if (!Py_IsInitialized()) {
        cbridge_log(LOG_INFO, "initializing Python");
        Py_InitializeEx(1);
        if (!Py_IsInitialized()) {
            cbridge_log(LOG_ERROR, "Python initialization failed");
            return -1;
        }
        PyEval_InitThreads();
        main_thread_state = PyEval_SaveThread();
        cbridge_log(LOG_INFO, "Python initialized in main thread");

        // Import the kvcached module while we have the GIL
        PyEval_RestoreThread(main_thread_state);
        PyObject* sys = PyImport_ImportModule("sys");
        if (sys) {
            PyObject* path = PyObject_GetAttrString(sys, "path");
            if (path) {
                PyObject* pPath = PyUnicode_FromString("/home/ztang23/kvcached");
                if (pPath) {
                    PyList_Append(path, pPath);
                    Py_DECREF(pPath);
                }
                Py_DECREF(path);
            }
            Py_DECREF(sys);
        }

        kvcached_module = PyImport_ImportModule("kvcached.integration.ollama.interfaces");
        if (!kvcached_module) {
            cbridge_log(LOG_ERROR, "failed to import kvcached module");
            PyErr_Print();
            main_thread_state = PyEval_SaveThread();
            return -1;
        }
        cbridge_log(LOG_INFO, "kvcached module imported successfully");
        main_thread_state = PyEval_SaveThread();
    }

    shutdown_requested = 0;
    current_message = NULL;
    thread_initialized = 0;

    cbridge_log(LOG_INFO, "creating Python thread");

    if (pthread_create(&python_thread, NULL, python_thread_func, NULL) != 0) {
        cbridge_log(LOG_ERROR, "failed to create thread");
        return -1;
    }

    thread_running = 1;

    // Wait for thread to initialize
    cbridge_log(LOG_DEBUG, "waiting for thread initialization");
    pthread_mutex_lock(&queue_mutex);
    while (!thread_initialized) {
        pthread_cond_wait(&init_cond, &queue_mutex);
    }
    pthread_mutex_unlock(&queue_mutex);

    cbridge_log(LOG_INFO, "Python thread fully initialized");

    return 0;
}

// Send a message to the Python thread and wait for response
int send_message(bridge_message_t* msg) {
    if (!thread_running) return -1;

    msg->processed = 0;  // Initialize processed flag

    pthread_mutex_lock(&queue_mutex);
    current_message = msg;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);

    // Wait for completion (polling since we can't use cond vars easily for response)
    while (!msg->processed) {
        usleep(1000); // 1ms
    }

    return msg->result;
}

// Initialize Python interpreter and import kvcached module
int kvcached_bridge_init() {
    return bridge_init();
}

// Call Python init_kvcached function
int kvcached_bridge_init_kvcached(const char* device, int async_sched) {
    cbridge_log(LOG_INFO, "init_kvcached called with device=%s, async_sched=%d", device, async_sched);

    // Initialize bridge if not already done
    if (kvcached_bridge_init() != 0) {
        cbridge_log(LOG_ERROR, "bridge_init failed");
        return -1;
    }

    bridge_message_t msg;
    msg.type = BRIDGE_OP_INIT;
    msg.data.init.device = device;
    msg.data.init.async_sched = async_sched;

    cbridge_log(LOG_DEBUG, "init_kvcached calling Python thread");
    int result = send_message(&msg);
    cbridge_log(LOG_INFO, "init_kvcached result=%d", result);
    return result;
}

// Call Python alloc_kv_cache function (Stage 2)
int kvcached_bridge_alloc_kv_cache(int num_blocks, int block_size, int head_num, int head_dim, int num_layers, const char* device) {
    cbridge_log(LOG_INFO, "alloc_kv_cache called: blocks=%d, head=(%d,%d), layers=%d, device=%s",
                num_blocks, head_num, head_dim, num_layers, device);

    bridge_message_t msg;
    msg.type = BRIDGE_OP_ALLOC_KV_CACHE;
    msg.data.alloc_cache.num_blocks = num_blocks;
    msg.data.alloc_cache.block_size = block_size;
    msg.data.alloc_cache.head_num = head_num;
    msg.data.alloc_cache.head_dim = head_dim;
    msg.data.alloc_cache.num_layers = num_layers;
    msg.data.alloc_cache.device = device;

    int result = send_message(&msg);
    cbridge_log(LOG_INFO, "alloc_kv_cache result=%d", result);
    return result;
}

// Call Python shutdown_kvcached function
int kvcached_bridge_shutdown_kvcached() {
    cbridge_log(LOG_INFO, "shutdown_kvcached called");

    bridge_message_t msg;
    msg.type = BRIDGE_OP_SHUTDOWN;

    int result = send_message(&msg);

    // Shutdown the thread
    shutdown_requested = 1;
    pthread_cond_broadcast(&queue_cond);
    if (thread_running) {
        pthread_join(python_thread, NULL);
        thread_running = 0;
    }

    return result;
}

// Call Python alloc_kv function (allocate blocks for a request)
long long* kvcached_bridge_alloc_kv(int num_blocks) {
    cbridge_log(LOG_DEBUG, "alloc_kv called for %d blocks", num_blocks);

    bridge_message_t msg;
    msg.type = BRIDGE_OP_ALLOC_KV_BRIDGE;
    msg.data.alloc_bridge.num_blocks = num_blocks;
    msg.result_blocks = NULL;

    int result = send_message(&msg);

    if (result == 0 && msg.result_blocks) {
        cbridge_log(LOG_DEBUG, "alloc_kv succeeded");
        return msg.result_blocks;
    } else {
        cbridge_log(LOG_WARN, "alloc_kv failed, falling back to dummy allocation");
        long long* dummy = (long long*)malloc(num_blocks * sizeof(long long));
        if (dummy) {
            for (int i = 0; i < num_blocks; i++) {
                dummy[i] = i + 1; // dummy block IDs: 1, 2, 3, ...
            }
        }
        return dummy;
    }
}

// Call Python free_kv function (free blocks for a request)
int kvcached_bridge_free_kv(long long* block_ids, int num_blocks) {
    if (!block_ids) {
        return -1;
    }

    cbridge_log(LOG_DEBUG, "free_kv called for %d blocks", num_blocks);

    bridge_message_t msg;
    msg.type = BRIDGE_OP_FREE_KV;
    msg.data.free.block_ids = block_ids;
    msg.data.free.num_blocks = num_blocks;

    return send_message(&msg);
}

// Set logging level at runtime
void kvcached_bridge_set_log_level(int level) {
    if (level >= LOG_DEBUG && level <= LOG_ERROR) {
        current_log_level = (log_level_t)level;
        cbridge_log(LOG_INFO, "Log level set to %d", level);
    }
}