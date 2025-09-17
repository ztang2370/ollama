package kvcache

import (
	"log/slog"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// NoOpCache is a cache implementation that satisfies the Cache interface
// but performs no actual caching operations. This is used when kvcached
// is enabled to prevent conflicts between native cache and kvcached memory management.
type NoOpCache struct{
	// Store batch information from StartForward to create appropriately sized tensors
	curBatchSize int
	curPositions []int32
	curSequences []int
	
	// Pre-created tensors to avoid context setup issues
	emptyKey   ml.Tensor
	emptyValue ml.Tensor
	emptyMask  ml.Tensor
}

func NewNoOpCache() *NoOpCache {
	return &NoOpCache{}
}

// SetLayer sets the active layer of the cache (no-op)
func (c *NoOpCache) SetLayer(layer int) {
	// No operation
}

// Get returns pre-created empty tensors to indicate no cached data
func (c *NoOpCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
    slog.Debug("🔵 NoOpCache.Get called", "batchSize", c.curBatchSize, "tensorsAvailable", c.emptyKey != nil)

    // Return pre-created tensors from Init/StartForward if available
    if c.emptyKey != nil && c.emptyValue != nil {
        slog.Debug("🔵 NoOpCache.Get returning pre-created empty tensors")
        return c.emptyKey, c.emptyValue, c.emptyMask
    }

    // As a fallback, attempt to create empty tensors from empty slices which
    // is more permissive in ggml than Zeros/Empty that require Input/Layer.
    var key, val, mask ml.Tensor
    defer func() {
        if r := recover(); r != nil {
            slog.Warn("🔵 NoOpCache.Get panic recovered, returning nil tensors", "panic", r)
            key, val, mask = nil, nil, nil
        }
    }()

    empty := []float32{}
    key = ctx.FromFloatSlice(empty, 0)
    val = ctx.FromFloatSlice(empty, 0)
    mask = ctx.FromFloatSlice(empty, 0)
    slog.Debug("🔵 NoOpCache.Get returning ctx-created empty slice tensors")
    return key, val, mask
}

// Put stores a batch of key and value in the cache (no-op)
func (c *NoOpCache) Put(ctx ml.Context, key, value ml.Tensor) {
	// No operation
}

// SetConfig controls optimizations (no-op)
func (c *NoOpCache) SetConfig(config ml.CacheConfig) {
	// No operation
}

// Init sets up runtime parameters and creates minimal tensor storage
func (c *NoOpCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
    // No-op: do not attempt tensor creation here; some backends require Input/Layer
    // to be set before any tensor creation. Tensors will be handled by native cache.
}

// Close closes the cache and frees resources (no-op)
func (c *NoOpCache) Close() {
	// No operation - no resources to free
}

// StartForward is called before the start of the model's forward pass
// Store batch information (tensors are pre-created in Init)
func (c *NoOpCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	slog.Debug("🔵 NoOpCache.StartForward called", "batchSize", len(batch.Positions), "reserve", reserve)
	// Store batch information for Get() method
	c.curBatchSize = len(batch.Positions)
	c.curPositions = batch.Positions
	c.curSequences = batch.Sequences
	
	slog.Debug("🔵 NoOpCache.StartForward completed successfully", "tensorsAvailable", c.emptyKey != nil)
	return nil
}

// CopyPrefix copies tokens in the range [0, len) from srcSeq to dstSeq (no-op)
func (c *NoOpCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
	// No operation
}

// CanResume returns true if the cache can continue with the next token (always true for no-op)
func (c *NoOpCache) CanResume(seq int, pos int32) bool {
	// Always return true since we don't actually track anything
	return true
}

// Remove deletes tokens in the range [beginIndex, endIndex) from seq (no-op)
func (c *NoOpCache) Remove(seq int, beginIndex, endIndex int32) error {
	// No operation - always succeed
	return nil
}

// NewNoOpWrapperCache creates a real WrapperCache but with NoOpCache as the underlying cache
// This ensures type compatibility while preventing actual cache operations
func NewNoOpWrapperCache() *WrapperCache {
	slog.Debug("🔵 Creating WrapperCache with NoOpCache underlying")
	// Create a real WrapperCache with NoOpCache as the underlying cache
	return NewWrapperCache(NewNoOpCache())
}
