package laya

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Build turns one input item into the state and battery to evaluate for it.
type Build[T any] func(T) (state any, questions Questions)

// Result pairs an input item with what came back for it. Exactly one of
// Response and Err is set.
type Result[T any] struct {
	// Item is the input this result belongs to.
	Item T
	// Response is the successful evaluation, or nil when Err is set.
	Response *Response
	// Err is why this item failed, or nil on success.
	Err error
}

// OK reports whether this item was evaluated successfully.
func (r Result[T]) OK() bool { return r.Err == nil && r.Response != nil }

// BatchStats summarises a whole batch.
type BatchStats struct {
	// Wall is how long the batch took end to end.
	Wall time.Duration
	// Latencies holds the per-request round trip times, in completion order.
	Latencies []time.Duration
	// InputTokens and OutputTokens are the batch totals. Laya generates no
	// text, so OutputTokens is zero.
	InputTokens  int
	OutputTokens int
	// Succeeded and Failed count the items.
	Succeeded int
	Failed    int
	// Checkpoints counts how many items each checkpoint answered, which is how
	// you see what the router actually did over a run.
	Checkpoints map[string]int
}

// Percentile returns the latency at q, where q runs from 0 to 1, by nearest rank.
func (s BatchStats) Percentile(q float64) time.Duration {
	if len(s.Latencies) == 0 {
		return 0
	}
	ordered := make([]time.Duration, len(s.Latencies))
	copy(ordered, s.Latencies)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	idx := int(q*float64(len(ordered)-1) + 0.5)
	return ordered[min(max(idx, 0), len(ordered)-1)]
}

// BatchOptions tunes [Batch].
type BatchOptions struct {
	// Concurrency is how many requests may be in flight. Defaults to 8.
	//
	// A sidecar runs one forward pass at a time whatever this says, so raising
	// it only helps against a server with more than one worker.
	Concurrency int
	// OnProgress, if set, is called after each item finishes with the number
	// done and the total. It is called from several goroutines, already
	// serialised, so it must not block for long.
	OnProgress func(done, total int)
	// StopOnError cancels the remaining work as soon as one item fails. By
	// default a batch runs to the end and reports failures per item, which is
	// what you want for a benchmark.
	StopOnError bool
}

// Batch evaluates many items, one request each, and returns the results in the
// same order as the input.
//
// Failures are reported per item rather than aborting the run, so a single bad
// sample cannot lose a long benchmark. Use StopOnError to change that.
func Batch[T any](
	ctx context.Context,
	c *Client,
	items []T,
	build Build[T],
	opts BatchOptions,
) ([]Result[T], BatchStats) {
	concurrency := opts.Concurrency
	if concurrency < 1 {
		concurrency = 8
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]Result[T], len(items))
	stats := BatchStats{
		Latencies:   make([]time.Duration, 0, len(items)),
		Checkpoints: map[string]int{},
	}

	var (
		mu    sync.Mutex
		wg    sync.WaitGroup
		done  int
		gate  = make(chan struct{}, concurrency)
		start = time.Now()
	)

	// record keeps the stats and the progress count covering every item,
	// including the ones cancelled before they ever ran.
	record := func(i int, result Result[T]) {
		mu.Lock()
		results[i] = result
		if result.OK() {
			stats.Succeeded++
			stats.Latencies = append(stats.Latencies, result.Response.Latency)
			stats.InputTokens += result.Response.Usage.InputTokens
			stats.OutputTokens += result.Response.Usage.OutputTokens
			if checkpoint := result.Response.Checkpoint(); checkpoint != "" {
				stats.Checkpoints[checkpoint]++
			}
		} else {
			stats.Failed++
		}
		done++
		progress, total := done, len(items)
		report := opts.OnProgress
		mu.Unlock()

		if report != nil {
			report(progress, total)
		}
		if !result.OK() && opts.StopOnError {
			cancel()
		}
	}

	for i, item := range items {
		wg.Add(1)
		go func(i int, item T) {
			defer wg.Done()

			select {
			case gate <- struct{}{}:
				defer func() { <-gate }()
			case <-ctx.Done():
				record(i, Result[T]{Item: item, Err: ctx.Err()})
				return
			}

			state, questions := build(item)
			resp, err := c.Ask(ctx, state, questions)
			record(i, Result[T]{Item: item, Response: resp, Err: err})
		}(i, item)
	}

	wg.Wait()
	stats.Wall = time.Since(start)
	return results, stats
}
