package laya

import (
	"errors"
	"strings"
	"testing"
)

func TestRouteByScriptAndLanguage(t *testing.T) {
	tests := []struct {
		name       string
		state      any
		wantModel  string
		wantScript string
		wantReason string
	}{
		{
			name:       "English prose stays on the English checkpoint",
			state:      map[string]any{"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
			wantModel:  ModelEnglish,
			wantScript: "latin",
			wantReason: "English Latin text",
		},
		{
			// The English checkpoint cannot read Devanagari at all: it scores
			// near random while reporting high confidence.
			name:       "Devanagari goes multilingual",
			state:      map[string]any{"body": "मुझे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
			wantModel:  ModelMultilingual,
			wantScript: "devanagari",
			wantReason: "non-Latin script",
		},
		{
			// Laya's own README uses this example for the Latin-but-not-English case.
			name:       "German is Latin script but not English",
			state:      map[string]any{"body": "Der Kunde wurde zweimal belastet"},
			wantModel:  ModelMultilingual,
			wantScript: "latin",
			wantReason: `language looks like "de"`,
		},
		{
			name:       "Korean goes multilingual",
			state:      "고객이 두 번 청구되었습니다",
			wantModel:  ModelMultilingual,
			wantScript: "hangul",
			wantReason: "non-Latin script",
		},
		{
			name:       "a state with no letters falls back to the default",
			state:      map[string]any{"invoice": "4411", "amount": 129.5},
			wantModel:  ModelEnglish,
			wantScript: "unknown",
			wantReason: "no letters detected",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decision := Route(tc.state, nil, RouteOptions{})

			if decision.Model != tc.wantModel {
				t.Errorf("model = %q, want %q (reason: %s)", decision.Model, tc.wantModel, decision.Reason)
			}
			if decision.Detection == nil {
				t.Fatal("detection = nil, want the language pass to be reported")
			}
			if decision.Detection.Script != tc.wantScript {
				t.Errorf("script = %q, want %q", decision.Detection.Script, tc.wantScript)
			}
			if !strings.Contains(decision.Reason, tc.wantReason) {
				t.Errorf("reason = %q, want it to mention %q", decision.Reason, tc.wantReason)
			}
			if decision.Repo != Models[tc.wantModel] {
				t.Errorf("repo = %q, want %q", decision.Repo, Models[tc.wantModel])
			}
		})
	}
}

func TestRoutePrecedence(t *testing.T) {
	hindi := "मुझे दो बार शुल्क लिया गया"

	tests := []struct {
		name      string
		opts      RouteOptions
		wantModel string
		wantWhy   string
	}{
		{"explicit model wins over detection", RouteOptions{Model: "en"}, ModelEnglish, "explicit model"},
		{"explicit task selects its checkpoint", RouteOptions{Task: TaskTypedDecisions}, ModelTypedDecisions, "explicit task"},
		{"explicit lang skips detection", RouteOptions{Lang: "en-GB"}, ModelEnglish, "explicit lang"},
		{"a non-English lang goes multilingual", RouteOptions{Lang: "fr"}, ModelMultilingual, "explicit lang"},
		{"lang guess hints english", RouteOptions{LangGuess: "en"}, ModelEnglish, "language guess"},
		{"lang guess hints multilingual", RouteOptions{LangGuess: "pt"}, ModelMultilingual, "language guess"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			decision := Route(hindi, nil, tc.opts)

			if decision.Model != tc.wantModel {
				t.Errorf("model = %q, want %q", decision.Model, tc.wantModel)
			}
			if !strings.Contains(decision.Reason, tc.wantWhy) {
				t.Errorf("reason = %q, want it to mention %q", decision.Reason, tc.wantWhy)
			}
			if decision.Detection != nil {
				t.Error("detection ran even though the decision was explicit")
			}
		})
	}
}

func TestWorkflowDetectionIsOptIn(t *testing.T) {
	battery := Questions{
		"action":      Options("What next?", "pay", "hold"),
		"category":    Options("Which category?", "billing", "technical"),
		"churn_risk":  Holds("Might they leave?"),
		"needs_human": Holds("Does this need a person?"),
		"urgency":     Levels("How urgent?", "low", "high"),
	}

	if got := MatchWorkflow(battery); got != "customer_service" {
		t.Fatalf("MatchWorkflow() = %q, want customer_service", got)
	}

	// A specialised checkpoint should never be a silent default.
	silent := Route("a support ticket in English", battery, RouteOptions{})
	if silent.Model != ModelEnglish {
		t.Errorf("model = %q, want the workflow to be ignored without opting in", silent.Model)
	}
	if silent.Workflow != "customer_service" {
		t.Errorf("workflow = %q, want it reported even when unused", silent.Workflow)
	}

	opted := Route("a support ticket in English", battery, RouteOptions{AutoTaskDetection: true})
	if opted.Model != ModelTypedDecisions {
		t.Errorf("model = %q, want %q once opted in", opted.Model, ModelTypedDecisions)
	}
}

func TestMatchWorkflowNeedsAnExactIDSet(t *testing.T) {
	// An unrelated battery that happens to contain "urgency" must not match.
	battery := Questions{
		"urgency": Levels("How urgent?", "low", "high"),
		"topic":   Options("About what?", "a", "b"),
	}

	if got := MatchWorkflow(battery); got != "" {
		t.Errorf("MatchWorkflow() = %q, want no match", got)
	}
}

func TestNormaliseModel(t *testing.T) {
	aliases := map[string]string{
		"en": ModelEnglish, "laya": ModelEnglish, "ENGLISH": ModelEnglish,
		"multi": ModelMultilingual, " ml ": ModelMultilingual,
		"typed_decisions": ModelTypedDecisions, "typed-decisions": ModelTypedDecisions,
	}
	for alias, want := range aliases {
		got, err := NormaliseModel(alias)
		if err != nil {
			t.Fatalf("NormaliseModel(%q): %v", alias, err)
		}
		if got != want {
			t.Errorf("NormaliseModel(%q) = %q, want %q", alias, got, want)
		}
	}

	if _, err := NormaliseModel("gpt-4"); !errors.Is(err, ErrUnknownModel) {
		t.Errorf("NormaliseModel(gpt-4) = %v, want ErrUnknownModel", err)
	}
}

func TestAnalyseReportsScriptMix(t *testing.T) {
	detection := Analyse("Hello — 你好世界你好世界")

	if detection.Script != "han" {
		t.Errorf("script = %q, want han", detection.Script)
	}
	if detection.IsEnglish {
		t.Error("is_english = true, want false for dominant Han text")
	}
	if detection.NonLatinFraction <= 0.5 {
		t.Errorf("non_latin_fraction = %v, want most letters outside Latin", detection.NonLatinFraction)
	}
	if detection.ScriptProfile["latin"] == 0 {
		t.Error("script_profile lost the Latin letters that are there")
	}
}

func TestShortStatesStayUndecided(t *testing.T) {
	// Fewer than four words is not enough to call a language, on purpose.
	detection := Analyse("Rechnung falsch")

	if detection.Language != "" {
		t.Errorf("language = %q, want undecided", detection.Language)
	}
	if !detection.IsEnglish {
		t.Error("is_english = false, want an undecided state to stay on the default")
	}
}

func TestStateTextWalksStructs(t *testing.T) {
	type ticket struct {
		Subject string   `json:"subject"`
		Tags    []string `json:"tags"`
		Reply   *ticket  `json:"reply,omitempty"`
	}

	text := StateText(ticket{
		Subject: "Duplicate charge",
		Tags:    []string{"billing"},
		Reply:   &ticket{Subject: "Refunded"},
	}, 4000)

	for _, want := range []string{"Duplicate charge", "billing", "Refunded"} {
		if !strings.Contains(text, want) {
			t.Errorf("StateText() = %q, want it to contain %q", text, want)
		}
	}
}

func TestStateTextTruncatesByRune(t *testing.T) {
	text := StateText(strings.Repeat("é", 50), 10)

	if got := len([]rune(text)); got != 10 {
		t.Errorf("StateText() kept %d runes, want 10", got)
	}
}

func TestIsEnglish(t *testing.T) {
	if !IsEnglish("Please refund the duplicate charge on this invoice") {
		t.Error("IsEnglish() = false for English prose")
	}
	if IsEnglish("मुझे दो बार शुल्क लिया गया") {
		t.Error("IsEnglish() = true for Devanagari")
	}
}
