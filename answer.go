package laya

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// Kind is the type of a question, and of the answer it produces.
type Kind string

// The three decision primitives.
const (
	KindNoul   Kind = "noul"
	KindChoice Kind = "choice"
	KindScore  Kind = "score"
)

// Confidence gate / abstention states reported on answers when min_confidence is configured.
const (
	// GatePassed indicates the confidence gate ran and this answer's confidence cleared it.
	GatePassed = "passed"
	// GateAbstained indicates the confidence gate ran and this answer's confidence fell below the threshold.
	GateAbstained = "abstained"
	// GateUnevaluated indicates the confidence gate ran but this answer carried no usable confidence.
	GateUnevaluated = "unevaluated"
)

// Answer is one typed decision returned by the model. The concrete types are
// [NoulAnswer], [ChoiceAnswer] and [ScoreAnswer].
//
// Typed output guarantees the shape of the result, not its truth.
type Answer interface {
	// Kind reports which of the three answer types this is.
	Kind() Kind
	// Confidence summarises how concentrated the distribution was, from 0 to 1 (normalized entropy).
	Confidence() float64
	// AnswerConfidence reports the calibrated max(p) confidence for this answer.
	AnswerConfidence() float64
	sealed()
}

// NoulAnswer is the calibrated probability that the answer to a [Noul] is yes.
type NoulAnswer struct {
	// Value runs from 0 (no) to 1 (yes).
	Value float64
	// Conf is max(Value, 1-Value): how far the model is from undecided.
	Conf float64
	// AnsConf is the calibrated answer_confidence (max(p)).
	AnsConf float64
	// Action is the model's auxiliary action head output.
	Action Action
	// LowConfidence indicates whether min_confidence triggered abstention on this answer.
	LowConfidence bool
	// Abstention reports "passed", "abstained", or "unevaluated" when a gate was run.
	Abstention string
	// AbstentionThreshold reports the threshold this answer was gated against.
	AbstentionThreshold *float64
}

// Kind implements [Answer].
func (NoulAnswer) Kind() Kind { return KindNoul }

// Confidence implements [Answer].
func (n NoulAnswer) Confidence() float64 { return n.Conf }

// AnswerConfidence implements [Answer].
func (n NoulAnswer) AnswerConfidence() float64 {
	if n.AnsConf != 0 {
		return n.AnsConf
	}
	return n.Conf
}

func (NoulAnswer) sealed() {}

// ChoiceAnswer is the option selected from a [Choice], with the full distribution.
type ChoiceAnswer struct {
	// Selected is the option with the highest probability.
	Selected string
	// Probabilities maps every option to its probability. They sum to 1.
	Probabilities map[string]float64
	// Conf summarises how concentrated Probabilities is, from 0 to 1. It says
	// how clearly one option beat the others, not whether acting is safe.
	Conf float64
	// AnsConf is the calibrated answer_confidence (max(p)).
	AnsConf float64
	// Action is the model's auxiliary action head output.
	Action Action
	// LowConfidence indicates whether min_confidence triggered abstention on this answer.
	LowConfidence bool
	// Abstention reports "passed", "abstained", or "unevaluated" when a gate was run.
	Abstention string
	// AbstentionThreshold reports the threshold this answer was gated against.
	AbstentionThreshold *float64
}

// Kind implements [Answer].
func (ChoiceAnswer) Kind() Kind { return KindChoice }

// Confidence implements [Answer].
func (c ChoiceAnswer) Confidence() float64 { return c.Conf }

// AnswerConfidence implements [Answer].
func (c ChoiceAnswer) AnswerConfidence() float64 {
	if c.AnsConf != 0 {
		return c.AnsConf
	}
	return c.Conf
}

func (ChoiceAnswer) sealed() {}

// Runners returns the options ordered from most to least probable.
func (c ChoiceAnswer) Runners() []string {
	options := make([]string, 0, len(c.Probabilities))
	for option := range c.Probabilities {
		options = append(options, option)
	}
	sort.Slice(options, func(i, j int) bool {
		if c.Probabilities[options[i]] != c.Probabilities[options[j]] {
			return c.Probabilities[options[i]] > c.Probabilities[options[j]]
		}
		return options[i] < options[j]
	})
	return options
}

// ScoreAnswer is a probability weighted position across the levels of a [Score].
type ScoreAnswer struct {
	// Value is the weighted position across the levels. It can land between two
	// levels, so 1.6 means the answer sits mostly at level 2.
	Value float64
	// Legend maps each level index to the description you supplied.
	Legend map[string]string
	// Probabilities maps each level index to its probability. They sum to 1.
	Probabilities map[string]float64
	// Conf summarises how concentrated Probabilities is, from 0 to 1.
	Conf float64
	// AnsConf is the calibrated answer_confidence (max(p)).
	AnsConf float64
	// Action is the model's auxiliary action head output.
	Action Action
	// LowConfidence indicates whether min_confidence triggered abstention on this answer.
	LowConfidence bool
	// Abstention reports "passed", "abstained", or "unevaluated" when a gate was run.
	Abstention string
	// AbstentionThreshold reports the threshold this answer was gated against.
	AbstentionThreshold *float64
}

// Kind implements [Answer].
func (ScoreAnswer) Kind() Kind { return KindScore }

// Confidence implements [Answer].
func (s ScoreAnswer) Confidence() float64 { return s.Conf }

// AnswerConfidence implements [Answer].
func (s ScoreAnswer) AnswerConfidence() float64 {
	if s.AnsConf != 0 {
		return s.AnsConf
	}
	return s.Conf
}

func (ScoreAnswer) sealed() {}

// Nearest returns the index and description of the closest whole level.
func (s ScoreAnswer) Nearest() (int, string) {
	level := int(math.Round(s.Value))
	return level, s.Legend[fmt.Sprint(level)]
}

// Action is the auxiliary head Laya reports beside every answer. It is a
// secondary signal from training, not a calibrated decision: branch on the
// answer itself.
type Action struct {
	// Probability is the action head's output for this question.
	Probability float64 `json:"act_probability"`
}

// Answers holds one answer per question id from a single request.
type Answers map[string]Answer

// ErrNoAnswer is returned when a question id is absent from a response.
type ErrNoAnswer struct{ ID string }

func (e *ErrNoAnswer) Error() string { return fmt.Sprintf("laya: no answer for question %q", e.ID) }

// ErrWrongKind is returned when an answer is read as the wrong type.
type ErrWrongKind struct {
	ID   string
	Want Kind
	Got  Kind
}

func (e *ErrWrongKind) Error() string {
	return fmt.Sprintf("laya: question %q answered with %s, not %s", e.ID, e.Got, e.Want)
}

// Noul reads the answer to a [Noul] question as a probability.
func (a Answers) Noul(id string) (float64, error) {
	answer, ok := a[id]
	if !ok {
		return 0, &ErrNoAnswer{ID: id}
	}
	noul, ok := answer.(NoulAnswer)
	if !ok {
		return 0, &ErrWrongKind{ID: id, Want: KindNoul, Got: answer.Kind()}
	}
	return noul.Value, nil
}

// Choice reads the answer to a [Choice] question.
func (a Answers) Choice(id string) (ChoiceAnswer, error) {
	answer, ok := a[id]
	if !ok {
		return ChoiceAnswer{}, &ErrNoAnswer{ID: id}
	}
	choice, ok := answer.(ChoiceAnswer)
	if !ok {
		return ChoiceAnswer{}, &ErrWrongKind{ID: id, Want: KindChoice, Got: answer.Kind()}
	}
	return choice, nil
}

// Score reads the answer to a [Score] question.
func (a Answers) Score(id string) (ScoreAnswer, error) {
	answer, ok := a[id]
	if !ok {
		return ScoreAnswer{}, &ErrNoAnswer{ID: id}
	}
	score, ok := answer.(ScoreAnswer)
	if !ok {
		return ScoreAnswer{}, &ErrWrongKind{ID: id, Want: KindScore, Got: answer.Kind()}
	}
	return score, nil
}

// MustNoul reads a [Noul] answer and returns 0 when it is missing or the wrong
// kind. Use it only when the battery is a literal in the same function, so a
// mismatch is a programming error you would have caught in testing.
func (a Answers) MustNoul(id string) float64 {
	value, err := a.Noul(id)
	if err != nil {
		return 0
	}
	return value
}

// wireAnswer is the untyped shape Laya returns, before it is narrowed.
type wireAnswer struct {
	Type                Kind               `json:"type"`
	Noul                float64            `json:"noul"`
	Choice              string             `json:"choice"`
	Score               float64            `json:"score"`
	Legend              map[string]string  `json:"legend"`
	Probabilities       map[string]float64 `json:"probabilities"`
	Confidence          float64            `json:"confidence"`
	AnswerConfidence    *float64           `json:"answer_confidence"`
	Action              Action             `json:"action"`
	LowConfidence       bool               `json:"low_confidence"`
	Abstention          string             `json:"abstention"`
	AbstentionThreshold *float64           `json:"abstention_threshold"`
}

// UnmarshalJSON implements [json.Unmarshaler], narrowing each answer to its type.
func (a *Answers) UnmarshalJSON(data []byte) error {
	var wire map[string]wireAnswer
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("laya: decoding answers: %w", err)
	}
	out := make(Answers, len(wire))
	for id, w := range wire {
		ansConf := w.Confidence
		if w.AnswerConfidence != nil {
			ansConf = *w.AnswerConfidence
		}
		switch w.Type {
		case KindNoul:
			out[id] = NoulAnswer{
				Value:               w.Noul,
				Conf:                w.Confidence,
				AnsConf:             ansConf,
				Action:              w.Action,
				LowConfidence:       w.LowConfidence,
				Abstention:          w.Abstention,
				AbstentionThreshold: w.AbstentionThreshold,
			}
		case KindChoice:
			out[id] = ChoiceAnswer{
				Selected:            w.Choice,
				Probabilities:       w.Probabilities,
				Conf:                w.Confidence,
				AnsConf:             ansConf,
				Action:              w.Action,
				LowConfidence:       w.LowConfidence,
				Abstention:          w.Abstention,
				AbstentionThreshold: w.AbstentionThreshold,
			}
		case KindScore:
			out[id] = ScoreAnswer{
				Value:               w.Score,
				Legend:              w.Legend,
				Probabilities:       w.Probabilities,
				Conf:                w.Confidence,
				AnsConf:             ansConf,
				Action:              w.Action,
				LowConfidence:       w.LowConfidence,
				Abstention:          w.Abstention,
				AbstentionThreshold: w.AbstentionThreshold,
			}
		default:
			return fmt.Errorf("laya: unknown answer type %q for question %q", w.Type, id)
		}
	}
	*a = out
	return nil
}

// sortedKeys returns the keys of m in sorted order.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
