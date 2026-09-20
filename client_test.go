package laya

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// predictJSON is a whole response from laya's Router.predict, routing included.
const predictJSON = `{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.94, "technical": 0.06},
      "confidence": 0.91,
      "action": {"act_probability": 0.8}
    },
    "churn_risk": {"type": "noul", "noul": 0.892, "confidence": 0.892, "action": {"act_probability": 0.6}}
  },
  "usage": {"input_tokens": 204, "output_tokens": 0},
  "routing": {
    "model": "english",
    "repo": "convaiinnovations/laya",
    "reason": "English Latin text",
    "detection": {"script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
                  "is_english": true, "non_latin_fraction": 0.0},
    "workflow": null
  }
}`

// battery is a small valid battery for tests that do not care what is asked.
func battery() Questions {
	return Questions{
		"department": OneOf("Which team?", map[string]string{"billing": "money", "technical": "bugs"}),
		"churn_risk": Holds("Might they leave?"),
	}
}

// newTestClient points a client at a stub server, and returns both.
func newTestClient(t *testing.T, handler http.HandlerFunc, opts ...Option) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	client, err := New(append([]Option{WithBaseURL(server.URL)}, opts...)...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func TestAskDecodesAWholeResponse(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/predict" {
			t.Errorf("path = %q, want /v1/predict", r.URL.Path)
		}
		fmt.Fprint(w, predictJSON)
	})

	resp, err := client.Ask(context.Background(), "I was billed twice", battery())
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if resp.Model != "laya-rl-agent" {
		t.Errorf("model = %q", resp.Model)
	}
	if resp.Checkpoint() != ModelEnglish {
		t.Errorf("checkpoint = %q, want %q", resp.Checkpoint(), ModelEnglish)
	}
	if resp.Routing.Reason != "English Latin text" {
		t.Errorf("routing reason = %q", resp.Routing.Reason)
	}
	if resp.Usage.InputTokens != 204 {
		t.Errorf("input tokens = %d, want 204", resp.Usage.InputTokens)
	}
	if resp.Latency <= 0 {
		t.Error("latency was not measured")
	}
	if churn := resp.Answers.MustNoul("churn_risk"); churn != 0.892 {
		t.Errorf("churn_risk = %v, want 0.892", churn)
	}
}

func TestAskSendsStateQuestionsAndModel(t *testing.T) {
	var body struct {
		State     map[string]any             `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
		Model     string                     `json:"model"`
		Lang      string                     `json:"lang"`
	}

	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request: %v", err)
		}
		fmt.Fprint(w, predictJSON)
	})

	state := map[string]any{"subject": "Duplicate charge", "body": "Refund it"}
	if _, err := client.Ask(context.Background(), state, battery(), UseModel("multi"), UseLang("de")); err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if body.State["subject"] != "Duplicate charge" {
		t.Errorf("state = %#v", body.State)
	}
	if len(body.Questions) != 2 {
		t.Errorf("sent %d questions, want 2", len(body.Questions))
	}
	if body.Model != ModelMultilingual {
		t.Errorf("model = %q, want the alias resolved to %q", body.Model, ModelMultilingual)
	}
	if body.Lang != "de" {
		t.Errorf("lang = %q, want de", body.Lang)
	}
}

func TestClientModelPinsEveryRequest(t *testing.T) {
	var sent string
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		var body Request
		_ = json.NewDecoder(r.Body).Decode(&body)
		sent = body.Model
		fmt.Fprint(w, predictJSON)
	}, WithModel("multilingual"))

	if _, err := client.Ask(context.Background(), "state", battery()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if sent != ModelMultilingual {
		t.Errorf("model = %q, want %q", sent, ModelMultilingual)
	}
}

func TestNewRejectsAnUnknownModel(t *testing.T) {
	if _, err := New(WithModel("gpt-4")); !errors.Is(err, ErrUnknownModel) {
		t.Errorf("New = %v, want ErrUnknownModel", err)
	}
}

func TestAMalformedBatteryNeverLeavesTheProcess(t *testing.T) {
	called := false
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		called = true
		fmt.Fprint(w, predictJSON)
	})

	_, err := client.Ask(context.Background(), "state", Questions{"a": Levels("How much?", "only one")})

	if !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("err = %v, want ErrInvalidRequest", err)
	}
	if called {
		t.Error("a battery that cannot be answered was still sent")
	}
}

func TestRetriesUntilTheServerRecovers(t *testing.T) {
	var attempts atomic.Int32
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, predictJSON)
	}, WithRetry(Retry{Attempts: 3, Base: time.Millisecond, Max: 5 * time.Millisecond}))

	if _, err := client.Ask(context.Background(), "state", battery()); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("made %d attempts, want 3", got)
	}
}

func TestAValidationFailureIsNotRetried(t *testing.T) {
	var attempts atomic.Int32
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprint(w, `{"error": "question 'topic' options exceed head_max_len=192"}`)
	}, WithRetry(Retry{Attempts: 4, Base: time.Millisecond}))

	_, err := client.Ask(context.Background(), "state", battery())

	if !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("err = %v, want it to match ErrInvalidRequest", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("made %d attempts, want 1: retrying cannot fix a malformed battery", got)
	}

	var serverErr *ServerError
	if !errors.As(err, &serverErr) {
		t.Fatalf("err = %v, want *ServerError", err)
	}
	if !strings.Contains(serverErr.Body, "head_max_len") {
		t.Errorf("body = %q, want the server's explanation kept", serverErr.Body)
	}
}

func TestNoServerIsReportedAsUnavailable(t *testing.T) {
	client, err := New(
		WithBaseURL("http://127.0.0.1:1"),
		WithRetry(Retry{Attempts: 1}),
		WithTimeout(time.Second),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = client.Ask(context.Background(), "state", battery())

	if !errors.Is(err, ErrUnavailable) {
		t.Errorf("err = %v, want ErrUnavailable", err)
	}
}

func TestRouteAsksTheRuntimeFirst(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/route" {
			t.Errorf("path = %q, want /v1/route", r.URL.Path)
		}
		fmt.Fprint(w, `{"model": "multilingual", "repo": "convaiinnovations/laya/multilingual",
		                "reason": "the server said so", "detection": null, "workflow": null}`)
	})

	decision, err := client.Route(context.Background(), "anything", battery())
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if decision.Model != ModelMultilingual || decision.Reason != "the server said so" {
		t.Errorf("decision = %#v, want the runtime's own answer", decision)
	}
}

func TestRouteFallsBackToTheLocalPort(t *testing.T) {
	client, err := New(WithBaseURL("http://127.0.0.1:1"), WithTimeout(time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	decision, err := client.Route(context.Background(), "मुझसे दो बार शुल्क लिया गया", battery())
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if decision.Model != ModelMultilingual {
		t.Errorf("model = %q, want the local router to answer when no runtime does", decision.Model)
	}
}

func TestWarmBuildsCheckpointsUpFront(t *testing.T) {
	var body struct {
		Models []string `json:"models"`
	}
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/warm" {
			t.Errorf("path = %q, want /v1/warm", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, `{"loaded": ["english"]}`)
	})

	if err := client.Warm(context.Background(), "en"); err != nil {
		t.Fatalf("Warm: %v", err)
	}
	if len(body.Models) != 1 || body.Models[0] != ModelEnglish {
		t.Errorf("models = %v, want the alias resolved to [english]", body.Models)
	}

	if err := client.Warm(context.Background(), "nonexistent"); !errors.Is(err, ErrUnknownModel) {
		t.Errorf("Warm = %v, want ErrUnknownModel", err)
	}
}

func TestContextCancellationStopsTheCall(t *testing.T) {
	release := make(chan struct{})
	defer close(release)

	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		<-release // a runtime that never answers
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	_, err := client.Ask(ctx, "state", battery())

	if err == nil {
		t.Fatal("Ask returned nil, want the cancellation reported")
	}
	// The call must give up on the deadline rather than working through its
	// retries first.
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Errorf("Ask took %v to notice a cancelled context", elapsed)
	}
}

func TestRetryDelayGrowsAndIsCapped(t *testing.T) {
	policy := Retry{Attempts: 5, Base: 100 * time.Millisecond, Max: 250 * time.Millisecond}

	if first, second := policy.delay(1), policy.delay(2); second <= first {
		t.Errorf("delays %v then %v, want the backoff to grow", first, second)
	}
	if capped := policy.delay(10); capped > policy.Max {
		t.Errorf("delay = %v, want it capped at %v", capped, policy.Max)
	}
}
