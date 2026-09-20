package laya

import (
	"encoding/json"
	"errors"
	"testing"
)

// marshalQuestion renders one question the way a request would.
func marshalQuestion(t *testing.T, q Question) map[string]any {
	t.Helper()
	encoded, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshalling %T: %v", q, err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("decoding %T: %v", q, err)
	}
	return decoded
}

func TestNoulMarshalsWithoutCriteria(t *testing.T) {
	wire := marshalQuestion(t, Holds("Does the user threaten to cancel?"))

	if wire["type"] != string(KindNoul) {
		t.Errorf("type = %v, want %s", wire["type"], KindNoul)
	}
	if _, ok := wire["criteria"]; ok {
		t.Error("criteria present, want it omitted so Laya uses its own true/false wording")
	}
}

func TestNoulMarshalsCriteria(t *testing.T) {
	wire := marshalQuestion(t, YesNo("Is this phishing?", "a scam", "legitimate"))

	criteria, ok := wire["criteria"].(map[string]any)
	if !ok {
		t.Fatalf("criteria = %#v, want an object", wire["criteria"])
	}
	if criteria["true"] != "a scam" || criteria["false"] != "legitimate" {
		t.Errorf("criteria = %#v", criteria)
	}
}

func TestChoiceSendsMissingDescriptionsAsNull(t *testing.T) {
	// An empty description must reach the model as null, so the option renders
	// as its bare name rather than "name: ".
	wire := marshalQuestion(t, Options("What is this about?", "coding", "billing"))

	criteria, ok := wire["criteria"].(map[string]any)
	if !ok {
		t.Fatalf("criteria = %#v, want an object", wire["criteria"])
	}
	for _, option := range []string{"coding", "billing"} {
		value, present := criteria[option]
		if !present {
			t.Fatalf("option %q missing from %#v", option, criteria)
		}
		if value != nil {
			t.Errorf("criteria[%q] = %#v, want null", option, value)
		}
	}
}

func TestScoreMarshalsOrderedLevels(t *testing.T) {
	wire := marshalQuestion(t, Levels("How urgent?", "calm", "soon", "now"))

	criteria, ok := wire["criteria"].([]any)
	if !ok {
		t.Fatalf("criteria = %#v, want an array", wire["criteria"])
	}
	want := []string{"calm", "soon", "now"}
	if len(criteria) != len(want) {
		t.Fatalf("got %d levels, want %d", len(criteria), len(want))
	}
	for i, level := range want {
		if criteria[i] != level {
			t.Errorf("level %d = %v, want %q", i, criteria[i], level)
		}
	}
}

func TestStructuredInstructionsSurvive(t *testing.T) {
	wire := marshalQuestion(t, Noul{Instructions: map[string]string{
		"question": "Is this a refund request?",
		"note":     "a duplicate charge counts",
	}})

	instructions, ok := wire["instructions"].(map[string]any)
	if !ok {
		t.Fatalf("instructions = %#v, want an object", wire["instructions"])
	}
	if instructions["note"] != "a duplicate charge counts" {
		t.Errorf("instructions = %#v", instructions)
	}
}

func TestQuestionsValidate(t *testing.T) {
	tests := []struct {
		name      string
		questions Questions
		wantErr   bool
	}{
		{"empty battery", Questions{}, true},
		{"nil question", Questions{"a": nil}, true},
		{"noul without instructions", Questions{"a": Noul{}}, true},
		{"choice with one option", Questions{"a": Options("Which?", "only")}, true},
		{"choice with a blank option", Questions{"a": OneOf("Which?", map[string]string{"": "x", "b": "y"})}, true},
		{"score with one level", Questions{"a": Levels("How much?", "some")}, true},
		{"score with an empty level", Questions{"a": Score{Instructions: "How much?", Criteria: []Content{"low", ""}}}, true},
		{"a valid battery", TriageQuestions(), false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.questions.Validate()
			if tc.wantErr {
				if err == nil {
					t.Fatal("Validate() = nil, want an error")
				}
				if !errors.Is(err, ErrInvalidRequest) {
					t.Errorf("Validate() = %v, want it to match ErrInvalidRequest", err)
				}
				return
			}
			if err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}

func TestPresetsAreValid(t *testing.T) {
	presets := map[string]Questions{
		"triage":     TriageQuestions(),
		"email":      EmailQuestions(nil),
		"guard":      GuardQuestions(),
		"moderation": ModerationQuestions(),
		"router":     RouterQuestions(),
	}
	for name, questions := range presets {
		t.Run(name, func(t *testing.T) {
			if err := questions.Validate(); err != nil {
				t.Fatalf("%s preset does not validate: %v", name, err)
			}
			if _, err := json.Marshal(questions); err != nil {
				t.Fatalf("%s preset does not marshal: %v", name, err)
			}
		})
	}
}

func TestEmailQuestionsTakeCustomCategories(t *testing.T) {
	questions := EmailQuestions(map[string]string{"legal": "contracts", "other": "everything else"})

	category, ok := questions["category"].(Choice)
	if !ok {
		t.Fatalf("category = %T, want Choice", questions["category"])
	}
	if _, present := category.Criteria["legal"]; !present {
		t.Errorf("criteria = %#v, want the supplied categories", category.Criteria)
	}
	if _, present := category.Criteria["billing"]; present {
		t.Error("default categories leaked into a custom set")
	}
}
