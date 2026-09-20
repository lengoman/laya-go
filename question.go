package laya

import (
	"encoding/json"
	"fmt"
)

// Question is one typed decision to ask about a state. The three
// implementations are [Noul], [Choice] and [Score]. The interface is closed:
// only this package can define a question type, because the model has exactly
// three decision heads.
type Question interface {
	// Kind reports the wire value of the question's "type" field.
	Kind() Kind
	// validate reports why the question cannot be answered as written.
	validate() error
	sealed()
}

// Content is anything that can be sent as instructions or as a criterion. A
// plain string covers most questions. A map or slice is useful when
// definitions, contrasts or examples make the meaning clearer; Laya renders
// structured content as compact JSON before it reaches the encoder.
type Content any

// Noul asks a yes or no question and returns the calibrated probability that
// the answer is yes. Use one Noul per label when several labels may apply at
// once.
//
// A Noul near 0.5 means the model finds yes and no about equally likely. It
// does not mean "medium intensity".
type Noul struct {
	// Instructions is the yes or no question to evaluate. Required.
	Instructions Content
	// Criteria optionally describes what a yes and a no mean. Supplying it
	// usually sharpens the answer, because it pins down the boundary. Left out,
	// Laya falls back to "yes, the statement holds" and "no, it does not".
	Criteria *NoulCriteria
}

// NoulCriteria describes the two ends of a [Noul].
type NoulCriteria struct {
	// True is what a probability near 1 means.
	True string `json:"true,omitempty"`
	// False is what a probability near 0 means.
	False string `json:"false,omitempty"`
}

// Kind implements [Question].
func (Noul) Kind() Kind { return KindNoul }
func (Noul) sealed()    {}

func (n Noul) validate() error {
	if isEmptyContent(n.Instructions) {
		return fmt.Errorf("noul: instructions are required")
	}
	return nil
}

// MarshalJSON implements [json.Marshaler].
func (n Noul) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         Kind          `json:"type"`
		Instructions Content       `json:"instructions"`
		Criteria     *NoulCriteria `json:"criteria,omitempty"`
	}{KindNoul, n.Instructions, n.Criteria})
}

// Choice picks exactly one option from a set you define, and returns the whole
// probability distribution over those options.
//
// Include a no-match option when none of the others may fit: the model cannot
// pick an option that is not in Criteria. Every option shares one token budget
// (head_max_len), so past roughly twenty options the descriptions start to
// blur into one another; see [Config] for raising the budget.
type Choice struct {
	// Instructions is what the model should decide. Required.
	Instructions Content
	// Criteria maps each option to a description of it. An empty description is
	// allowed when the option name speaks for itself. At least two are required.
	Criteria map[string]string
}

// Kind implements [Question].
func (Choice) Kind() Kind { return KindChoice }
func (Choice) sealed()    {}

func (c Choice) validate() error {
	if isEmptyContent(c.Instructions) {
		return fmt.Errorf("choice: instructions are required")
	}
	if len(c.Criteria) < 2 {
		return fmt.Errorf("choice: needs at least 2 options, has %d", len(c.Criteria))
	}
	for option := range c.Criteria {
		if option == "" {
			return fmt.Errorf("choice: an option name is empty")
		}
	}
	return nil
}

// MarshalJSON implements [json.Marshaler].
func (c Choice) MarshalJSON() ([]byte, error) {
	// An absent description must reach the model as JSON null, not as "", so
	// the option renders as its bare name rather than "name: ".
	criteria := make(map[string]any, len(c.Criteria))
	for option, description := range c.Criteria {
		if description == "" {
			criteria[option] = nil
			continue
		}
		criteria[option] = description
	}
	return json.Marshal(struct {
		Type         Kind           `json:"type"`
		Instructions Content        `json:"instructions"`
		Criteria     map[string]any `json:"criteria"`
	}{KindChoice, c.Instructions, criteria})
}

// Score rates a state against ordered levels and returns a probability
// weighted position across them, so the result can land between two levels.
//
// Levels must be ordered, lowest first, and each must describe a concrete
// situation that stands on its own. Score is Laya's weakest primitive; where a
// threshold is what you actually need, a [Noul] is usually sharper.
type Score struct {
	// Instructions is what the model should rate. Required.
	Instructions Content
	// Criteria is the ordered list of level descriptions, lowest first. At
	// least two are required.
	Criteria []Content
}

// Kind implements [Question].
func (Score) Kind() Kind { return KindScore }
func (Score) sealed()    {}

func (s Score) validate() error {
	if isEmptyContent(s.Instructions) {
		return fmt.Errorf("score: instructions are required")
	}
	if len(s.Criteria) < 2 {
		return fmt.Errorf("score: needs at least 2 levels, has %d", len(s.Criteria))
	}
	for i, level := range s.Criteria {
		if isEmptyContent(level) {
			return fmt.Errorf("score: level %d is empty", i)
		}
	}
	return nil
}

// MarshalJSON implements [json.Marshaler].
func (s Score) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type         Kind      `json:"type"`
		Instructions Content   `json:"instructions"`
		Criteria     []Content `json:"criteria"`
	}{KindScore, s.Instructions, s.Criteria})
}

// Questions is a battery of questions keyed by an id you choose. The answer to
// each comes back under the same id.
//
// Ids are for your code only. They are not shown to the model, so each question
// must carry its full meaning. The one exception is the router's opt-in
// workflow detection, which matches the four typed-decisions batteries by their
// id sets; see [MatchWorkflow].
type Questions map[string]Question

// Validate reports the first question that cannot be answered as written. The
// client calls it before every request, so a malformed battery fails in
// microseconds rather than after a checkpoint load.
func (q Questions) Validate() error {
	if len(q) == 0 {
		return fmt.Errorf("%w: no questions", ErrInvalidRequest)
	}
	for _, id := range sortedKeys(q) {
		question := q[id]
		if question == nil {
			return fmt.Errorf("%w: question %q is nil", ErrInvalidRequest, id)
		}
		if err := question.validate(); err != nil {
			return fmt.Errorf("%w: question %q: %w", ErrInvalidRequest, id, err)
		}
	}
	return nil
}

// IDs returns the question ids in sorted order.
func (q Questions) IDs() []string { return sortedKeys(q) }

// YesNo builds a [Noul] with both criteria filled in.
func YesNo(instructions, whenTrue, whenFalse string) Noul {
	return Noul{
		Instructions: instructions,
		Criteria:     &NoulCriteria{True: whenTrue, False: whenFalse},
	}
}

// Holds builds a [Noul] from the question alone, leaving Laya's default
// true/false descriptions in place.
func Holds(instructions string) Noul {
	return Noul{Instructions: instructions}
}

// OneOf builds a [Choice] from a map of option to description.
func OneOf(instructions string, options map[string]string) Choice {
	return Choice{Instructions: instructions, Criteria: options}
}

// Options builds a [Choice] from bare option names, for when the names speak
// for themselves.
func Options(instructions string, options ...string) Choice {
	criteria := make(map[string]string, len(options))
	for _, option := range options {
		criteria[option] = ""
	}
	return Choice{Instructions: instructions, Criteria: criteria}
}

// Levels builds a [Score] from ordered level descriptions, lowest first.
func Levels(instructions string, levels ...string) Score {
	criteria := make([]Content, len(levels))
	for i, level := range levels {
		criteria[i] = level
	}
	return Score{Instructions: instructions, Criteria: criteria}
}

// isEmptyContent reports whether content would reach the model as nothing at all.
func isEmptyContent(c Content) bool {
	switch v := c.(type) {
	case nil:
		return true
	case string:
		return v == ""
	}
	return false
}
