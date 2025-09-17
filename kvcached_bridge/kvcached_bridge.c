#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

// Global Python module reference and thread state
static PyObject* kvcached_module = NULL;
static PyThreadState* main_thread_state = NULL;

// Synchronous bridge using dedicated thread
typedef struct {
    int type;  // 0=init, 1=alloc_kv_cache, 2=alloc_kv_bridge, 3=free_kv, 4=shutdown
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

    fprintf(stderr, "C_BRIDGE: Python thread started\n");

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
            case 0: { // init_kvcached
                fprintf(stderr, "C_BRIDGE: Python thread processing init message\n");
                PyEval_RestoreThread(main_thread_state);

                // kvcached_module is already imported in main thread
                fprintf(stderr, "C_BRIDGE: kvcached_module = %p\n", kvcached_module);

                if (kvcached_module) {
                    // Call init_kvcached
                    fprintf(stderr, "C_BRIDGE: Calling init_kvcached function\n");
                    PyObject* pFunc = PyObject_GetAttrString(kvcached_module, "init_kvcached");
                    fprintf(stderr, "C_BRIDGE: pFunc = %p\n", pFunc);
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject* pArgs = PyTuple_New(5);
                        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(0));
                        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(1));
                        PyTuple_SetItem(pArgs, 2, PyBool_FromLong(0));
                        PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(msg->data.init.device));
                        PyTuple_SetItem(pArgs, 4, PyBool_FromLong(msg->data.init.async_sched));

                        fprintf(stderr, "C_BRIDGE: Calling Python function\n");
                        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                        fprintf(stderr, "C_BRIDGE: pResult = %p\n", pResult);
                        if (pResult) {
                            fprintf(stderr, "C_BRIDGE: init_kvcached succeeded\n");
                            msg->result = 0;
                            Py_DECREF(pResult);
                        } else {
                            fprintf(stderr, "C_BRIDGE: init_kvcached failed\n");
                            msg->result = -1;
                            PyErr_Print();
                        }
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        fprintf(stderr, "C_BRIDGE: Cannot find init_kvcached function\n");
                        msg->result = -1;
                    }
                } else {
                    fprintf(stderr, "C_BRIDGE: Failed to import kvcached module\n");
                    msg->result = -1;
                    PyErr_Print();
                }

                main_thread_state = PyEval_SaveThread();
                fprintf(stderr, "C_BRIDGE: Finished processing init message, result=%d\n", msg->result);
                msg->processed = 1;
                break;
            }
            case 1: { // alloc_kv_cache
                fprintf(stderr, "C_BRIDGE: Python thread processing alloc_kv_cache message\n");
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
                            fprintf(stderr, "C_BRIDGE: alloc_kv_cache succeeded\n");
                            msg->result = 0;
                            Py_DECREF(pResult);
                        } else {
                            fprintf(stderr, "C_BRIDGE: alloc_kv_cache failed\n");
                            msg->result = -1;
                            PyErr_Print();
                        }
                        Py_DECREF(pArgs);
                        Py_DECREF(pFunc);
                    } else {
                        fprintf(stderr, "C_BRIDGE: Cannot find alloc_kv_cache function\n");
                        msg->result = -1;
                    }
                } else {
                    fprintf(stderr, "C_BRIDGE: kvcached module not available\n");
                    msg->result = -1;
                }

                main_thread_state = PyEval_SaveThread();
                msg->processed = 1;
                break;
            }
            case 2: { // alloc_kv_bridge
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
            case 3: { // free_kv
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
            case 4: { // shutdown
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

    fprintf(stderr, "C_BRIDGE: Python thread exiting\n");
    return NULL;
}

// Initialize the synchronous bridge
int bridge_init() {
    fprintf(stderr, "C_BRIDGE: bridge_init called\n");

    if (thread_running) {
        fprintf(stderr, "C_BRIDGE: thread already running\n");
        return 0;
    }

    // Initialize Python in the main thread
    if (!Py_IsInitialized()) {
        fprintf(stderr, "C_BRIDGE: initializing Python\n");
        Py_InitializeEx(1);
        if (!Py_IsInitialized()) {
            fprintf(stderr, "C_BRIDGE: Python initialization failed\n");
            return -1;
        }
        PyEval_InitThreads();
        main_thread_state = PyEval_SaveThread();
        fprintf(stderr, "C_BRIDGE: Python initialized in main thread\n");

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
            fprintf(stderr, "C_BRIDGE: failed to import kvcached module\n");
            PyErr_Print();
            main_thread_state = PyEval_SaveThread();
            return -1;
        }
        fprintf(stderr, "C_BRIDGE: kvcached module imported successfully\n");
        main_thread_state = PyEval_SaveThread();
    }

    shutdown_requested = 0;
    current_message = NULL;
    thread_initialized = 0;

    fprintf(stderr, "C_BRIDGE: creating Python thread\n");

    if (pthread_create(&python_thread, NULL, python_thread_func, NULL) != 0) {
        fprintf(stderr, "C_BRIDGE: failed to create thread\n");
        return -1;
    }

    thread_running = 1;

    // Wait for thread to initialize
    fprintf(stderr, "C_BRIDGE: waiting for thread initialization\n");
    pthread_mutex_lock(&queue_mutex);
    while (!thread_initialized) {
        pthread_cond_wait(&init_cond, &queue_mutex);
    }
    pthread_mutex_unlock(&queue_mutex);

    fprintf(stderr, "C_BRIDGE: Python thread fully initialized\n");

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
    fprintf(stderr, "C_BRIDGE: init_kvcached called with device=%s, async_sched=%d\n", device, async_sched);

    // Initialize bridge if not already done
    if (kvcached_bridge_init() != 0) {
        fprintf(stderr, "C_BRIDGE: bridge_init failed\n");
        return -1;
    }

    bridge_message_t msg;
    msg.type = 0; // init
    msg.data.init.device = device;
    msg.data.init.async_sched = async_sched;

    fprintf(stderr, "C_BRIDGE: init_kvcached calling Python thread\n");
    int result = send_message(&msg);
    fprintf(stderr, "C_BRIDGE: init_kvcached result=%d\n", result);
    return result;
}

// Call Python alloc_kv_cache function (Stage 2)
int kvcached_bridge_alloc_kv_cache(int num_blocks, int block_size, int head_num, int head_dim, int num_layers, const char* device) {
    fprintf(stderr, "C_BRIDGE: alloc_kv_cache called with num_blocks=%d, block_size=%d, head_num=%d, head_dim=%d, num_layers=%d, device=%s\n", 
            num_blocks, block_size, head_num, head_dim, num_layers, device);

    bridge_message_t msg;
    msg.type = 1; // alloc_kv_cache
    msg.data.alloc_cache.num_blocks = num_blocks;
    msg.data.alloc_cache.block_size = block_size;
    msg.data.alloc_cache.head_num = head_num;
    msg.data.alloc_cache.head_dim = head_dim;
    msg.data.alloc_cache.num_layers = num_layers;
    msg.data.alloc_cache.device = device;

    int result = send_message(&msg);
    fprintf(stderr, "C_BRIDGE: alloc_kv_cache result=%d\n", result);
    return result;
}

// Call Python shutdown_kvcached function
int kvcached_bridge_shutdown_kvcached() {
    fprintf(stderr, "C_BRIDGE: shutdown_kvcached called\n");

    bridge_message_t msg;
    msg.type = 4; // shutdown

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
    fprintf(stderr, "C_BRIDGE: alloc_kv called\n");

    bridge_message_t msg;
    msg.type = 2; // alloc_kv_bridge
    msg.data.alloc_bridge.num_blocks = num_blocks;
    msg.result_blocks = NULL;

    int result = send_message(&msg);

    if (result == 0 && msg.result_blocks) {
        fprintf(stderr, "C_BRIDGE: alloc_kv succeeded\n");
        return msg.result_blocks;
    } else {
        fprintf(stderr, "C_BRIDGE: alloc_kv failed, falling back to dummy allocation\n");
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

    fprintf(stderr, "C_BRIDGE: free_kv called\n");

    bridge_message_t msg;
    msg.type = 3; // free_kv
    msg.data.free.block_ids = block_ids;
    msg.data.free.num_blocks = num_blocks;

    return send_message(&msg);
}