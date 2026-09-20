package laya

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// answersJSON is the shape laya's Agent.system_one returns for one of each type.
const answersJSON = `{
  "department": {
    "type": "choice",
    "choice": "billing",
    "probabilities": {"billing": 0.94, "technical": 0.04, "sales": 0.02},
    "confidence": 0.9123,
    "action": {"act_probability": 0.77}
  },
  "urgency": {
    "type": "score",
    "score": 1.84,
    "legend": {"0": "not urgent", "1": "soon", "2": "critical deadline"},
    "probabilities": {"0": 0.05, "1": 0.06, "2": 0.89},
    "confidence": 0.8412,
    "action": {"act_probability": 0.51}
  },
  "churn_risk": {
    "type": "noul",
    "noul": 0.892,
    "confidence": 0.892,
    "action": {"act_probability": 0.42}
  }
}`

func decodeAnswers(t *testing.T) Answers {
	t.Helper()
	var answers Answers
	if err := json.Unmarshal([]byte(answersJSON), &answers); err != nil {
		t.Fatalf("decoding answers: %v", err)
	}
	return answers
}

func TestAnswersNarrowToTheirType(t *testing.T) {
	answers := decodeAnswers(t)

	churn, err := answers.Noul("churn_risk")
	if err != nil {
		t.Fatalf("Noul: %v", err)
	}
	if churn != 0.892 {
		t.Errorf("churn_risk = %v, want 0.892", churn)
	}

	department, err := answers.Choice("department")
	if err != nil {
		t.Fatalf("Choice: %v", err)
	}
	if department.Selected != "billing" {
		t.Errorf("selected = %q, want billing", department.Selected)
	}
	if department.Conf != 0.9123 {
		t.Errorf("confidence = %v, want 0.9123", department.Conf)
	}
	if department.Action.Probability != 0.77 {
		t.Errorf("act_probability = %v, want 0.77", department.Action.Probability)
	}

	urgency, err := answers.Score("urgency")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if urgency.Value != 1.84 {
		t.Errorf("score = %v, want 1.84", urgency.Value)
	}
	if level, label := urgency.Nearest(); level != 2 || label != "critical deadline" {
		t.Errorf("Nearest() = %d, %q, want 2, critical deadline", level, label)
	}
}

func TestReadingAnAnswerAsTheWrongKind(t *testing.T) {
	answers := decodeAnswers(t)

	_, err := answers.Noul("department")

	var wrong *ErrWrongKind
	if !errors.As(err, &wrong) {
		t.Fatalf("err = %v, want *ErrWrongKind", err)
	}
	if wrong.Want != KindNoul || wrong.Got != KindChoice {
		t.Errorf("want %s, got %s", wrong.Want, wrong.Got)
	}
}

func TestReadingAMissingAnswer(t *testing.T) {
	answers := decodeAnswers(t)

	_, err := answers.Score("nonexistent")

	var missing *ErrNoAnswer
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want *ErrNoAnswer", err)
	}
	if missing.ID != "nonexistent" {
		t.Errorf("ID = %q, want nonexistent", missing.ID)
	}
	if got := answers.MustNoul("nonexistent"); got != 0 {
		t.Errorf("MustNoul on a missing id = %v, want 0", got)
	}
}

func TestUnknownAnswerTypeIsAnError(t *testing.T) {
	var answers Answers
	err := json.Unmarshal([]byte(`{"a": {"type": "vibes"}}`), &answers)
	if err == nil {
		t.Fatal("decoding an unknown answer type succeeded, want an error")
	}
}

func TestChoiceRunnersRankOptions(t *testing.T) {
	answers := decodeAnswers(t)
	department, _ := answers.Choice("department")

	want := []string{"billing", "technical", "sales"}
	got := department.Runners()
	if len(got) != len(want) {
		t.Fatalf("Runners() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Runners() = %v, want %v", got, want)
		}
	}
}

func TestChoiceRunnersBreakTiesByName(t *testing.T) {
	answer := ChoiceAnswer{Probabilities: map[string]float64{"b": 0.5, "a": 0.5}}

	if got := answer.Runners(); got[0] != "a" {
		t.Errorf("Runners() = %v, want a first on a tie", got)
	}
}

func TestScoreNearestRoundsToALevel(t *testing.T) {
	answer := ScoreAnswer{Value: 1.49, Legend: map[string]string{"0": "low", "1": "mid", "2": "high"}}

	if level, label := answer.Nearest(); level != 1 || label != "mid" {
		t.Errorf("Nearest() = %d, %q, want 1, mid", level, label)
	}
}

func TestAnswerKindsAndConfidence(t *testing.T) {
	answers := decodeAnswers(t)

	kinds := map[string]Kind{"department": KindChoice, "urgency": KindScore, "churn_risk": KindNoul}
	for id, want := range kinds {
		answer := answers[id]
		if answer.Kind() != want {
			t.Errorf("%s kind = %s, want %s", id, answer.Kind(), want)
		}
		if confidence := answer.Confidence(); confidence <= 0 || confidence > 1 {
			t.Errorf("%s confidence = %v, want it in (0, 1]", id, confidence)
		}
	}
}

func TestProbabilitiesSumToOne(t *testing.T) {
	answers := decodeAnswers(t)
	department, _ := answers.Choice("department")

	total := 0.0
	for _, probability := range department.Probabilities {
		total += probability
	}
	if math.Abs(total-1) > 1e-9 {
		t.Errorf("probabilities sum to %v, want 1", total)
	}
}
