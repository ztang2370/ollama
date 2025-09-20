package ollamarunner

/*
#cgo CFLAGS: -I../../kvcached_bridge
#cgo LDFLAGS: -L../../kvcached_bridge -lkvcached_bridge
#include <stdlib.h>
#include "kvcached_bridge.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"
	"unsafe"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// Three Stage kvcached Integration (Replaces Native Ollama Cache):
// Stage 1 (System Startup): Call init_kvcached() in server.go before load()
// Stage 2 (Model Loading): Call alloc_kv_cache() during allocModel()
// Stage 3 (Request Processing): Call alloc_kv_bridge()/free_kv_bridge() as needed
//
// Memory Management:
// - When kvcached is enabled: Native cache disabled, kvcached handles all memory
// - When kvcached is disabled: Fall back to native Ollama cache management

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	// kvcached enabled flag
	kvcachedEnabled bool

	cache kvcache.Cache
}

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, batchSize int, multiUserCache bool, kvcachedEnabled bool) (*InputCache, error) {
	slog.Debug("NewInputCache called", "kvCacheType", kvCacheType, "kvSize", kvSize, "numSlots", numSlots, "batchSize", batchSize, "kvcachedEnabled", kvcachedEnabled)

	numCtx := kvSize / int32(numSlots)
	slog.Debug("Calculated numCtx", "numCtx", numCtx)

	if numCtx < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)
	slog.Debug("Created cache slots", "numSlots", numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

    // Choose cache implementation
    var cache kvcache.Cache = nil
    if kvcachedEnabled {
        // For correctness, use the model's native WrapperCache so attention gets valid K/V tensors.
        // kvcached will still manage block memory separately (Stage 3) without interfering here.
        slog.Info("kvcached enabled - using native WrapperCache for model attention")
        cache = model.Config().Cache
        if cache != nil {
            kvcachedCapacity := int(numCtx)
            slog.Debug("Initializing native cache (kvcached mode)", "kvCacheType", kvCacheTypeFromStr(kvCacheType), "numSlots", numSlots, "numCtx", numCtx, "kvcachedCapacity", kvcachedCapacity, "batchSize", batchSize)
            cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, kvcachedCapacity, batchSize)
        } else {
            slog.Debug("Model has no native cache - disabling cache")
        }
    } else {
        // Native (non-kvcached) path
        slog.Info("Using native Ollama cache management")
        cache = model.Config().Cache
        if cache != nil {
            slog.Debug("Initializing native cache", "kvCacheType", kvCacheTypeFromStr(kvCacheType), "numSlots", numSlots, "numCtx", numCtx, "batchSize", batchSize)
            cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, int(numCtx), batchSize)
        } else {
            slog.Debug("Model has no native cache - disabling cache")
        }
    }

	return &InputCache{
		numCtx:          numCtx,
		enabled:         cache != nil || kvcachedEnabled,
		slots:           slots,
		multiUserCache:  multiUserCache,
		kvcachedEnabled: kvcachedEnabled,
		cache:           cache,
	}, nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	default:
		return ml.DTypeF16
	}
}

func (c *InputCache) Close() {
	if c == nil {
		return
	}

	if c.cache != nil {
		c.cache.Close()
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time

	// kvcached blocks associated with this cache slot
	// These blocks persist across sequence completions for conversation continuity
	// and are only freed when the cache slot is actually evicted
	kvCacheBlocks []int32
}

func (c *InputCache) LoadCacheSlot(prompt []input.Input) (*InputCacheSlot, []input.Input, error) {
	slog.Debug("LoadCacheSlot called", "promptLen", len(prompt), "enabled", c.enabled, "cache", c.cache != nil, "kvcachedEnabled", c.kvcachedEnabled)

	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	// Memory management: choose between native cache and kvcached
	if c.kvcachedEnabled {
		// kvcached memory management (primary mode)
		slog.Debug("Using kvcached memory management")
		// Check if we need to reset cache due to conversation mismatch
		if numPast == 0 && len(slot.kvCacheBlocks) > 0 {
			// No prefix match - free old blocks and allocate new ones
			slog.Debug("Conversation mismatch - freeing old kvcached blocks for slot", "slot", slot.Id)
			c.freeSlotKvcachedBlocks(slot)
		}

		// Dynamic allocation with prefix caching preservation
		if numPast == 0 {
			// New conversation - free old blocks and allocate fresh
			slog.Debug("No prefix match - allocating fresh kvcached blocks", "slot", slot.Id, "promptLen", len(prompt))
			c.ensureKvcachedBlocks(slot, len(prompt))
		} else {
			// Prefix match - clean up tensor cache to match prefix, then allocate blocks
			if c.cache != nil {
				err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
				if err != nil {
					// Some models don't support partial erasure
					err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
					if err != nil {
						return nil, nil, err
					}
					numPast = 0
				}
			}
			slog.Debug("Prefix match found - cleaned tensor cache, allocating kvcached blocks", "slot", slot.Id, "numPast", numPast, "promptLen", len(prompt), "existingBlocks", len(slot.kvCacheBlocks))
			c.ensureKvcachedBlocks(slot, len(prompt))
		}
	} else {
		// Native Ollama cache management (fallback mode)
		slog.Debug("Using native cache management")
		if c.cache != nil {
			if numPast > 0 && !c.cache.CanResume(slot.Id, numPast) {
				numPast = 0
			}

			err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
			if err != nil {
				// Some models don't support partial erasure
				err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
				if err != nil {
					return nil, nil, err
				}
				numPast = 0
			}
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
		
		// Free kvcached blocks when slot is actually evicted
		c.freeSlotKvcachedBlocks(oldestSlot)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if !c.kvcachedEnabled && c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []input.Input, b []input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// TODO(jessegross): If we need to reprocess the inputs we should ensure that
// we don't split up a SameBatch
func (c *InputCache) ShiftDiscard(inputLen int32, numKeep int32) int32 {
	targetFree := (c.numCtx - numKeep) / 2
	targetFree = max(targetFree, 1)

	currentFree := c.numCtx - inputLen
	discard := targetFree - currentFree

	if discard < 0 {
		discard = 0
	}

	return discard
}

type ErrReprocessInputs struct {
	Inputs []input.Input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(inputLen, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.kvcachedEnabled {
		// For kvcached mode: preserve prefix caching by adjusting block allocation dynamically
		slog.Debug("Shifting kvcached blocks for slot", "id", slot.Id, "oldBlocks", len(slot.kvCacheBlocks))

		// Calculate how many blocks we need for remaining tokens after shift
		remainingTokens := numKeep + inputLen - (numKeep + discard)
		if remainingTokens > 0 {
			// Calculate needed blocks for remaining tokens
			blockSize := int32(32) // Standard kvcached block size
			promptTokens := int32(remainingTokens)
			estimatedResponseTokens := c.numCtx / 32  // Conservative estimate
			conversationBuffer := int32(0)
			totalEstimatedTokens := promptTokens + estimatedResponseTokens + conversationBuffer
			neededBlocks := (totalEstimatedTokens + blockSize - 1) / blockSize

			// Cap at max blocks
			maxBlocks := c.numCtx / blockSize
			if neededBlocks > maxBlocks {
				neededBlocks = maxBlocks
			}
			if neededBlocks < 1 {
				neededBlocks = 1
			}

			currentBlocks := int32(len(slot.kvCacheBlocks))

			if currentBlocks > neededBlocks {
				// Need to free excess blocks
				blocksToFree := currentBlocks - neededBlocks
				slog.Debug("Freeing excess kvcached blocks", "slotId", slot.Id, "current", currentBlocks, "needed", neededBlocks, "freeing", blocksToFree)

				// Free excess blocks from the end
				excessBlocks := slot.kvCacheBlocks[neededBlocks:]
				slot.kvCacheBlocks = slot.kvCacheBlocks[:neededBlocks]

				if len(excessBlocks) > 0 {
					// Convert Go slice to C array
					blockIds := make([]C.longlong, len(excessBlocks))
					for i, blockId := range excessBlocks {
						blockIds[i] = C.longlong(blockId)
					}

					// Free excess blocks
					result := C.kvcached_bridge_free_kv(&blockIds[0], C.int(len(excessBlocks)))
					if result != 0 {
						slog.Warn("Failed to free excess kvcached blocks", "slot", slot.Id, "blocks", len(excessBlocks))
					} else {
						slog.Debug("Freed excess kvcached blocks", "slot", slot.Id, "freed", len(excessBlocks))
					}
				}
			} else if currentBlocks < neededBlocks {
				// Need to allocate additional blocks (preserve existing)
				blocksToAllocate := neededBlocks - currentBlocks
				slog.Debug("Growing kvcached blocks during shift", "slotId", slot.Id, "current", currentBlocks, "needed", neededBlocks, "allocating", blocksToAllocate)

				// Allocate additional blocks
				blockIdsPtr := C.kvcached_bridge_alloc_kv(C.int(blocksToAllocate))
				if blockIdsPtr != nil {
					blockIdsSlice := (*[1 << 30]C.longlong)(unsafe.Pointer(blockIdsPtr))[:blocksToAllocate:blocksToAllocate]

					// Append new block IDs to existing list
					newBlocks := make([]int32, blocksToAllocate)
					for i := 0; i < int(blocksToAllocate); i++ {
						newBlocks[i] = int32(blockIdsSlice[i])
					}
					slot.kvCacheBlocks = append(slot.kvCacheBlocks, newBlocks...)
					C.free(unsafe.Pointer(blockIdsPtr))

					slog.Debug("Grew kvcached blocks during shift", "slot", slot.Id, "totalBlocks", len(slot.kvCacheBlocks), "addedBlocks", blocksToAllocate)
				} else {
					slog.Warn("Failed to allocate additional kvcached blocks during shift", "slot", slot.Id, "requested", blocksToAllocate)
				}
			} else {
				// Current allocation is sufficient, preserve prefix caching
				slog.Debug("Sufficient kvcached blocks already allocated during shift", "slotId", slot.Id, "current", currentBlocks, "needed", neededBlocks)
			}
		} else {
			// No remaining tokens, free all blocks
			slog.Debug("No remaining tokens after shift, freeing all kvcached blocks", "slot", slot.Id)
			c.freeSlotKvcachedBlocks(slot)
		}

		// In kvcached mode, we still need to update the tensor cache bookkeeping
		// to synchronize cell ownership with the shifted sequence positions.
		// kvcached manages the actual GPU memory, but the tensor cache tracks which
		// cells belong to which sequences for attention operations.
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			slot.Inputs = []input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	} else if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			slot.Inputs = []input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}

// freeSlotKvcachedBlocks frees kvcached blocks associated with a cache slot during eviction
func (c *InputCache) freeSlotKvcachedBlocks(slot *InputCacheSlot) {
	if len(slot.kvCacheBlocks) > 0 {
		// Convert Go slice to C array
		numBlocks := len(slot.kvCacheBlocks)
		blockIds := make([]C.longlong, numBlocks)
		for i, blockId := range slot.kvCacheBlocks {
			blockIds[i] = C.longlong(blockId)
		}

		// Stage 3: Free blocks using kvcached bridge during slot eviction
		result := C.kvcached_bridge_free_kv(&blockIds[0], C.int(numBlocks))
		if result == 0 {
			slog.Debug("Stage 3: Freed kvcached blocks during slot eviction",
				"slot", slot.Id, "blocks", numBlocks)
		} else {
			slog.Warn("Stage 3: Failed to free kvcached blocks during slot eviction",
				"slot", slot.Id, "blocks", numBlocks)
		}

		slot.kvCacheBlocks = nil
	}
}

// ensureKvcachedBlocks allocates or grows kvcached blocks for a cache slot as needed
func (c *InputCache) ensureKvcachedBlocks(slot *InputCacheSlot, promptLen int) {
	slog.Debug("ensureKvcachedBlocks called", "slotId", slot.Id, "currentBlocks", len(slot.kvCacheBlocks), "promptLen", promptLen)

	// Calculate how many blocks we need for this request
	blockSize := int32(32) // Standard kvcached block size
	promptTokens := int32(promptLen)
	estimatedResponseTokens := c.numCtx / 32  // Conservative estimate
	conversationBuffer := int32(0)
	totalEstimatedTokens := promptTokens + estimatedResponseTokens + conversationBuffer
	neededBlocks := (totalEstimatedTokens + blockSize - 1) / blockSize

	// Cap at max blocks
	maxBlocks := c.numCtx / blockSize
	if neededBlocks > maxBlocks {
		neededBlocks = maxBlocks
	}
	if neededBlocks < 1 {
		neededBlocks = 1
	}

	currentBlocks := int32(len(slot.kvCacheBlocks))

	if currentBlocks < neededBlocks {
		// Need to allocate additional blocks
		blocksToAllocate := neededBlocks - currentBlocks
		slog.Debug("Growing kvcached blocks", "slotId", slot.Id, "current", currentBlocks, "needed", neededBlocks, "allocating", blocksToAllocate)

		// Allocate additional blocks
		blockIdsPtr := C.kvcached_bridge_alloc_kv(C.int(blocksToAllocate))
		if blockIdsPtr != nil {
			blockIdsSlice := (*[1 << 30]C.longlong)(unsafe.Pointer(blockIdsPtr))[:blocksToAllocate:blocksToAllocate]

			// Append new block IDs to existing list
			newBlocks := make([]int32, blocksToAllocate)
			for i := 0; i < int(blocksToAllocate); i++ {
				newBlocks[i] = int32(blockIdsSlice[i])
			}
			slot.kvCacheBlocks = append(slot.kvCacheBlocks, newBlocks...)
			C.free(unsafe.Pointer(blockIdsPtr))

			slog.Debug("Grew kvcached blocks for cache slot", "slot", slot.Id, "totalBlocks", len(slot.kvCacheBlocks), "addedBlocks", blocksToAllocate)
		} else {
			slog.Warn("Failed to allocate additional kvcached blocks", "slot", slot.Id, "requested", blocksToAllocate)
		}
	} else {
		slog.Debug("Sufficient kvcached blocks already allocated", "slotId", slot.Id, "current", currentBlocks, "needed", neededBlocks)
	}
}
