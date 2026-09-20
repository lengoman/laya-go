package laya

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// batchServer answers every item, failing the ones whose state says to.
func batchServer(t *testing.T, inFlight *atomic.Int32, peak *atomic.Int32) *Client {
	t.Helper()
	return newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		now := inFlight.Add(1)
		for {
			high := peak.Load()
			if now <= high || peak.CompareAndSwap(high, now) {
				break
			}
		}
		defer inFlight.Add(-1)

		var body Request
		_ = json.NewDecoder(r.Body).Decode(&body)
		state, _ := body.State.(string)

		time.Sleep(5 * time.Millisecond)
		if strings.HasPrefix(state, "bad") {
			w.WriteHeader(http.StatusUnprocessableEntity)
			fmt.Fprint(w, `{"error": "malformed"}`)
			return
		}
		fmt.Fprint(w, predictJSON)
	}, WithRetry(Retry{Attempts: 1}))
}

func TestBatchKeepsInputOrderAndReportsFailuresPerItem(t *testing.T) {
	var inFlight, peak atomic.Int32
	client := batchServer(t, &inFlight, &peak)

	items := []string{"one", "bad-two", "three", "bad-four", "five"}
	results, stats := Batch(context.Background(), client, items,
		func(item string) (any, Questions) { return item, battery() },
		BatchOptions{Concurrency: 3},
	)

	if len(results) != len(items) {
		t.Fatalf("got %d results, want %d", len(results), len(items))
	}
	for i, result := range results {
		if result.Item != items[i] {
			t.Errorf("result %d is for %q, want %q", i, result.Item, items[i])
		}
		wantOK := !strings.HasPrefix(items[i], "bad")
		if result.OK() != wantOK {
			t.Errorf("result %d OK = %v, want %v (err: %v)", i, result.OK(), wantOK, result.Err)
		}
	}

	if stats.Succeeded != 3 || stats.Failed != 2 {
		t.Errorf("stats = %d ok, %d failed, want 3 and 2", stats.Succeeded, stats.Failed)
	}
	if stats.InputTokens != 3*204 {
		t.Errorf("input tokens = %d, want %d", stats.InputTokens, 3*204)
	}
	if stats.Checkpoints[ModelEnglish] != 3 {
		t.Errorf("checkpoints = %v, want three answered by english", stats.Checkpoints)
	}
	if stats.Wall <= 0 || stats.Percentile(0.95) <= 0 {
		t.Errorf("stats did not record timings: wall %v, p95 %v", stats.Wall, stats.Percentile(0.95))
	}
}

func TestBatchRespectsConcurrency(t *testing.T) {
	var inFlight, peak atomic.Int32
	client := batchServer(t, &inFlight, &peak)

	items := make([]string, 20)
	for i := range items {
		items[i] = fmt.Sprintf("item-%d", i)
	}

	Batch(context.Background(), client, items,
		func(item string) (any, Questions) { return item, battery() },
		BatchOptions{Concurrency: 4},
	)

	if got := peak.Load(); got > 4 {
		t.Errorf("peak in flight = %d, want at most 4", got)
	}
	if got := peak.Load(); got < 2 {
		t.Errorf("peak in flight = %d, want the work to actually overlap", got)
	}
}

func TestBatchReportsProgress(t *testing.T) {
	var inFlight, peak atomic.Int32
	client := batchServer(t, &inFlight, &peak)

	items := []string{"a", "b", "c"}
	var last atomic.Int32
	Batch(context.Background(), client, items,
		func(item string) (any, Questions) { return item, battery() },
		BatchOptions{Concurrency: 1, OnProgress: func(done, total int) {
			if total != len(items) {
				t.Errorf("total = %d, want %d", total, len(items))
			}
			last.Store(int32(done))
		}},
	)

	if got := last.Load(); got != int32(len(items)) {
		t.Errorf("last progress = %d, want %d", got, len(items))
	}
}

func TestBatchStopOnError(t *testing.T) {
	var inFlight, peak atomic.Int32
	client := batchServer(t, &inFlight, &peak)

	items := make([]string, 12)
	for i := range items {
		items[i] = fmt.Sprintf("bad-%d", i)
	}

	results, stats := Batch(context.Background(), client, items,
		func(item string) (any, Questions) { return item, battery() },
		BatchOptions{Concurrency: 1, StopOnError: true},
	)

	if stats.Failed != len(items) {
		t.Errorf("stats = %d failed, want every item accounted for", stats.Failed)
	}

	cancelled := 0
	for _, result := range results {
		if errors.Is(result.Err, context.Canceled) {
			cancelled++
		}
	}
	if cancelled == 0 {
		t.Error("the batch ran to the end, want the rest cancelled after the first failure")
	}
}

func TestBatchPercentileOnAnEmptyRun(t *testing.T) {
	if got := (BatchStats{}).Percentile(0.95); got != 0 {
		t.Errorf("Percentile on no data = %v, want 0", got)
	}
}

func TestBatchPercentileByNearestRank(t *testing.T) {
	stats := BatchStats{Latencies: []time.Duration{
		50 * time.Millisecond, 10 * time.Millisecond, 30 * time.Millisecond,
		20 * time.Millisecond, 40 * time.Millisecond,
	}}

	if got := stats.Percentile(0); got != 10*time.Millisecond {
		t.Errorf("p0 = %v, want 10ms", got)
	}
	if got := stats.Percentile(1); got != 50*time.Millisecond {
		t.Errorf("p100 = %v, want 50ms", got)
	}
	if got := stats.Percentile(0.5); got != 30*time.Millisecond {
		t.Errorf("p50 = %v, want 30ms", got)
	}
}
