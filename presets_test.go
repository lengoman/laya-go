package laya

import (
	"testing"
)

func TestStateFieldPresets(t *testing.T) {
	tests := []struct {
		name      string
		questions Questions
		want      string
	}{
		{"triage", TriageQuestions(), "message"},
		{"email", EmailQuestions(nil), "body"},
		{"guard", GuardQuestions(), "prompt"},
		{"moderation", ModerationQuestions(), "post"},
		{"router", RouterQuestions(), "request"},
		{"no field", Questions{"q": Holds("Is this valid?")}, ""},
		{"mixed fields", Questions{
			"q1": Holds("Check `message`."),
			"q2": Holds("Check `prompt`."),
		}, ""},
		{"same field across multiple questions", Questions{
			"q1": Holds("Check `data` for issues."),
			"q2": OneOf("Pick type in `data`", map[string]string{"a": "1", "b": "2"}),
		}, "data"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := StateField(tc.questions)
			if got != tc.want {
				t.Errorf("StateField() = %q, want %q", got, tc.want)
			}
		})
	}
}
