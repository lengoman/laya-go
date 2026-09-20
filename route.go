package laya

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

// The three checkpoints Laya ships.
const (
	// ModelEnglish is ModernBERT-large, 421M parameters, 512 tokens. Best on
	// English, and near random off it.
	ModelEnglish = "english"
	// ModelMultilingual is mmBERT-base, 322M parameters, 1024 tokens, 100+
	// languages, and about twice as fast.
	ModelMultilingual = "multilingual"
	// ModelTypedDecisions is ModernBERT-large fine-tuned on the four
	// typed-decisions workflows. It is never selected automatically unless you
	// opt in, because it is specialised.
	ModelTypedDecisions = "typed-decisions"

	// TaskTypedDecisions selects [ModelTypedDecisions] by task rather than by name.
	TaskTypedDecisions = "typed_decisions"

	// BundleRepo holds all three checkpoints; only the requested subfolder is
	// downloaded.
	BundleRepo = "convaiinnovations/laya"
)

// Models maps each checkpoint to the Hugging Face repo it is downloaded from.
var Models = map[string]string{
	ModelEnglish:        BundleRepo,
	ModelMultilingual:   BundleRepo + "/multilingual",
	ModelTypedDecisions: BundleRepo + "/typed-decisions",
}

// modelAliases are the names people are likely to type.
var modelAliases = map[string]string{
	"en": ModelEnglish, "laya": ModelEnglish, "default": ModelEnglish,
	"multi": ModelMultilingual, "ml": ModelMultilingual,
	"laya-multilingual": ModelMultilingual,
	"typed":             ModelTypedDecisions, "typed_decisions": ModelTypedDecisions,
	"laya-typed-decisions": ModelTypedDecisions, "decisions": ModelTypedDecisions,
}

// NormaliseModel resolves a checkpoint name or alias to its canonical name.
func NormaliseModel(name string) (string, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if alias, ok := modelAliases[key]; ok {
		key = alias
	}
	if _, ok := Models[key]; !ok {
		return "", fmt.Errorf("%w %q; choose one of %s", ErrUnknownModel, name,
			strings.Join([]string{ModelEnglish, ModelMultilingual, ModelTypedDecisions}, ", "))
	}
	return key, nil
}

// workflows are the question-id signatures of the four typed-decisions
// batteries. A match is exact, so an unrelated battery that happens to contain
// "urgency" is never captured.
var workflows = map[string][]string{
	"agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
	"customer_service":          {"action", "category", "churn_risk", "needs_human", "urgency"},
	"invoice_processing":        {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
	"security_incidents":        {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}

// MatchWorkflow names the typed-decisions workflow whose question ids these
// are, or "" when they are not one of the four.
func MatchWorkflow(questions Questions) string {
	ids := questions.IDs()
	for _, name := range sortedKeys(workflows) {
		signature := workflows[name]
		if len(ids) != len(signature) {
			continue
		}
		match := true
		for i, id := range ids {
			if id != signature[i] {
				match = false
				break
			}
		}
		if match {
			return name
		}
	}
	return ""
}

// RouteDecision is the routing outcome: which checkpoint, why, and what was
// detected. It is the "routing" key of a [Response].
type RouteDecision struct {
	// Model is the chosen checkpoint: "english", "multilingual" or
	// "typed-decisions".
	Model string `json:"model"`
	// Repo is the Hugging Face repo, and subfolder, it is loaded from.
	Repo string `json:"repo"`
	// Reason is the decision in words, fit to be logged next to the answer.
	Reason string `json:"reason"`
	// Detection is what the language pass found, or nil when the decision was
	// made without one.
	Detection *Detection `json:"detection"`
	// Workflow names the matched typed-decisions battery, when one matched.
	Workflow string `json:"workflow"`
}

func (d RouteDecision) String() string {
	return fmt.Sprintf("%s (%s)", d.Model, d.Reason)
}

// Detection is what [Analyse] found in a state.
type Detection struct {
	// Script is the dominant script: "latin", "devanagari", "han", and so on,
	// or "unknown" when the state holds no letters.
	Script string `json:"script"`
	// ScriptProfile is the fraction of letters belonging to each script found.
	ScriptProfile map[string]float64 `json:"script_profile"`
	// Language is a best-effort code for Latin-script text, or "" when
	// undecided. Short states are usually undecided on purpose.
	Language string `json:"language"`
	// IsEnglish reports whether the English checkpoint can be expected to read
	// this state.
	IsEnglish bool `json:"is_english"`
	// NonLatinFraction is the share of letters outside the Latin script.
	NonLatinFraction float64 `json:"non_latin_fraction"`
}

// RouteOptions overrides parts of a routing decision.
type RouteOptions struct {
	// Model pins the checkpoint, and beats everything else.
	Model string
	// Task names a workflow, and beats detection.
	Task string
	// Lang states the language, skipping detection.
	Lang string
	// Default is the checkpoint used when a state holds no letters. Empty
	// means [ModelEnglish].
	Default string
	// AutoTaskDetection opts in to selecting the typed-decisions checkpoint
	// when the question ids match one of its four workflows. Off by default,
	// because a specialised checkpoint should not be a silent default.
	AutoTaskDetection bool
}

// Route decides which checkpoint should answer, without loading or running
// anything. It is a port of Laya's own router, so a Go service can log or act
// on the decision before it pays for a call, and a test can assert on it with
// no Python in sight.
//
// Precedence: explicit Model, then Task, then a detected workflow when opted
// in, then explicit Lang, then the detected script and language, then the
// default.
//
// Script detection is exact. The Latin-script language guess is a
// stopword-and-diacritic heuristic and is explicitly best-effort: pass Lang
// when you already know the language.
func Route(state any, questions Questions, opts RouteOptions) RouteDecision {
	fallback := opts.Default
	if fallback == "" {
		fallback = ModelEnglish
	}

	decide := func(model, reason string, detection *Detection, workflow string) RouteDecision {
		name, err := NormaliseModel(model)
		if err != nil {
			name = ModelEnglish
		}
		return RouteDecision{
			Model:     name,
			Repo:      Models[name],
			Reason:    reason,
			Detection: detection,
			Workflow:  workflow,
		}
	}

	if opts.Model != "" {
		return decide(opts.Model, fmt.Sprintf("explicit model=%q", opts.Model), nil, "")
	}
	if opts.Task != "" {
		model := opts.Task
		if strings.ReplaceAll(strings.ToLower(opts.Task), "-", "_") == TaskTypedDecisions {
			model = ModelTypedDecisions
		}
		return decide(model, fmt.Sprintf("explicit task=%q", opts.Task), nil, "")
	}

	workflow := MatchWorkflow(questions)
	if workflow != "" && opts.AutoTaskDetection {
		return decide(ModelTypedDecisions,
			fmt.Sprintf("question ids match the %q typed-decisions workflow", workflow), nil, workflow)
	}

	if opts.Lang != "" {
		model := ModelMultilingual
		switch strings.SplitN(strings.ToLower(opts.Lang), "-", 2)[0] {
		case "en", "eng", "english":
			model = ModelEnglish
		}
		return decide(model, fmt.Sprintf("explicit lang=%q", opts.Lang), nil, workflow)
	}

	detection := Analyse(state)
	switch {
	case detection.Script == "unknown":
		return decide(fallback,
			fmt.Sprintf("no letters detected in state; using default (%s)", fallback), &detection, workflow)
	case detection.Script != "latin":
		return decide(ModelMultilingual, fmt.Sprintf(
			"non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it",
			detection.Script, 100*detection.NonLatinFraction), &detection, workflow)
	case !detection.IsEnglish:
		return decide(ModelMultilingual,
			fmt.Sprintf("Latin script but language looks like %q, not English", detection.Language),
			&detection, workflow)
	}
	return decide(ModelEnglish, "English Latin text", &detection, workflow)
}

// IsEnglish reports whether the English checkpoint can be expected to read
// this state.
func IsEnglish(state any) bool { return Analyse(state).IsEnglish }

// Analyse reports the script, and for Latin script the likely language, of a
// state. This is the signal Laya routes on, and it costs microseconds: the
// English checkpoint does not degrade gently off English, it collapses, and it
// stays confident while doing so, so the decision has to be made before the
// forward pass rather than from the model's own confidence.
func Analyse(state any) Detection {
	text := StateText(state, 4000)
	profile, order := scriptProfile(text)
	script := dominantScript(profile, order)

	nonLatin := 0.0
	if len(profile) > 0 {
		nonLatin = round4(1.0 - profile["latin"])
	}

	switch script {
	case "unknown":
		return Detection{Script: "unknown", ScriptProfile: profile, IsEnglish: true}
	case "latin":
		language := guessLatinLanguage(text)
		return Detection{
			Script:           "latin",
			ScriptProfile:    profile,
			Language:         language,
			IsEnglish:        language == "" || language == "en",
			NonLatinFraction: nonLatin,
		}
	}
	return Detection{Script: script, ScriptProfile: profile, NonLatinFraction: nonLatin}
}

// StateText flattens a state into the text detection reads. Keys are ignored,
// because they are usually English even when the content is not.
func StateText(state any, maxChars int) string {
	text := strings.Join(collectStrings(normalise(state), 0), " ")
	if maxChars > 0 {
		runes := []rune(text)
		if len(runes) > maxChars {
			text = string(runes[:maxChars])
		}
	}
	return text
}

// normalise turns any state into the string, map and slice tree that a JSON
// state is, so a struct reads the same way the runtime will read it.
func normalise(state any) any {
	switch state.(type) {
	case nil, string, map[string]any, []any:
		return state
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		return nil
	}
	var decoded any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		return nil
	}
	return decoded
}

// collectStrings walks the string leaves of a state. Map keys are visited in
// sorted order, which JSON object order does not guarantee, so that the text
// detection sees is at least stable from one call to the next.
func collectStrings(state any, depth int) []string {
	if depth > 6 || state == nil {
		return nil
	}
	switch value := state.(type) {
	case string:
		return []string{value}
	case map[string]any:
		var out []string
		for _, key := range sortedKeys(value) {
			out = append(out, collectStrings(value[key], depth+1)...)
		}
		return out
	case []any:
		var out []string
		for _, item := range value {
			out = append(out, collectStrings(item, depth+1)...)
		}
		return out
	}
	return nil
}

// scriptRange is one Unicode block belonging to a script.
type scriptRange struct {
	name   string
	blocks [][2]rune
}

// scriptRanges are the blocks the English checkpoint, on a 50k English BPE
// vocabulary, cannot read. Order matters: the first block that contains a rune
// claims it.
var scriptRanges = []scriptRange{
	{"greek", [][2]rune{{0x0370, 0x03FF}, {0x1F00, 0x1FFF}}},
	{"cyrillic", [][2]rune{{0x0400, 0x052F}, {0x2DE0, 0x2DFF}, {0xA640, 0xA69F}}},
	{"hebrew", [][2]rune{{0x0590, 0x05FF}}},
	{"arabic", [][2]rune{{0x0600, 0x06FF}, {0x0750, 0x077F}, {0x08A0, 0x08FF}, {0xFB50, 0xFDFF}, {0xFE70, 0xFEFF}}},
	{"devanagari", [][2]rune{{0x0900, 0x097F}, {0xA8E0, 0xA8FF}}},
	{"bengali", [][2]rune{{0x0980, 0x09FF}}},
	{"gurmukhi", [][2]rune{{0x0A00, 0x0A7F}}},
	{"gujarati", [][2]rune{{0x0A80, 0x0AFF}}},
	{"oriya", [][2]rune{{0x0B00, 0x0B7F}}},
	{"tamil", [][2]rune{{0x0B80, 0x0BFF}}},
	{"telugu", [][2]rune{{0x0C00, 0x0C7F}}},
	{"kannada", [][2]rune{{0x0C80, 0x0CFF}}},
	{"malayalam", [][2]rune{{0x0D00, 0x0D7F}}},
	{"sinhala", [][2]rune{{0x0D80, 0x0DFF}}},
	{"thai", [][2]rune{{0x0E00, 0x0E7F}}},
	{"lao", [][2]rune{{0x0E80, 0x0EFF}}},
	{"tibetan", [][2]rune{{0x0F00, 0x0FFF}}},
	{"myanmar", [][2]rune{{0x1000, 0x109F}}},
	{"georgian", [][2]rune{{0x10A0, 0x10FF}}},
	{"ethiopic", [][2]rune{{0x1200, 0x137F}}},
	{"khmer", [][2]rune{{0x1780, 0x17FF}}},
	{"hangul", [][2]rune{{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xAC00, 0xD7AF}}},
	{"kana", [][2]rune{{0x3040, 0x309F}, {0x30A0, 0x30FF}, {0x31F0, 0x31FF}}},
	{"han", [][2]rune{{0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}}},
}

// scriptOf names the script of one letter, or "" when no block claims it.
func scriptOf(r rune) string {
	if r < 0x0250 || (r >= 0x1E00 && r <= 0x1EFF) { // Latin and Latin Extended Additional
		return "latin"
	}
	for _, script := range scriptRanges {
		for _, block := range script.blocks {
			if r >= block[0] && r <= block[1] {
				return script.name
			}
		}
	}
	return ""
}

// scriptProfile returns the fraction of letters belonging to each script, and
// the order the scripts were first seen in, which decides ties.
func scriptProfile(text string) (map[string]float64, []string) {
	counts := map[string]int{}
	order := []string{}
	total := 0
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		name := scriptOf(r)
		if name == "" {
			continue
		}
		if _, seen := counts[name]; !seen {
			order = append(order, name)
		}
		counts[name]++
		total++
	}
	if total == 0 {
		return map[string]float64{}, order
	}
	// Latin is counted last, so it loses ties to a script that appeared first.
	sort.SliceStable(order, func(i, j int) bool { return order[j] == "latin" && order[i] != "latin" })
	profile := make(map[string]float64, len(counts))
	for name, count := range counts {
		profile[name] = float64(count) / float64(total)
	}
	return profile, order
}

// dominantScript is the script holding the most letters, or "unknown".
func dominantScript(profile map[string]float64, order []string) string {
	best, bestShare := "unknown", 0.0
	for _, name := range order {
		if profile[name] > bestShare {
			best, bestShare = name, profile[name]
		}
	}
	return best
}

// stopwords are function words per language. Latin-script languages overlap
// heavily (de, la, le, un, e, que), so each hit is weighted and a margin is
// required before calling something non-English.
var stopwords = []struct {
	lang  string
	words []string
}{
	{"en", []string{"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
		"this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
		"will", "can", "would", "there", "their", "what", "which", "please", "we", "i"}},
	{"fr", []string{"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
		"pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"}},
	{"de", []string{"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
		"auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"}},
	{"es", []string{"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
		"pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"}},
	{"pt", []string{"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
		"dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"}},
	{"it", []string{"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
		"questo", "questa", "anche", "come", "più", "sono", "nella", "alla"}},
	{"nl", []string{"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
		"door", "maar", "ook", "worden", "deze", "naar", "wordt"}},
}

var stopwordSets = func() []map[string]bool {
	sets := make([]map[string]bool, len(stopwords))
	for i, entry := range stopwords {
		set := make(map[string]bool, len(entry.words))
		for _, word := range entry.words {
			set[word] = true
		}
		sets[i] = set
	}
	return sets
}()

var nonEnglishDiacritics = func() map[rune]bool {
	set := map[rune]bool{}
	for _, r := range "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö" {
		set[r] = true
	}
	return set
}()

// guessLatinLanguage is a best-effort language code for Latin-script text, or
// "" when undecided. A non-English language must beat English by a margin, so
// ordinary English is never misrouted; short inputs stay undecided on purpose.
func guessLatinLanguage(text string) string {
	words := letterWords(strings.ToLower(text))
	if len(words) < 4 {
		return ""
	}

	scores := make([]int, len(stopwords))
	for _, word := range words {
		for i, set := range stopwordSets {
			if set[word] {
				scores[i]++
			}
		}
	}

	lowered := strings.ToLower(text)
	diacritics, runes := 0, 0
	for _, r := range lowered {
		runes++
		if nonEnglishDiacritics[r] {
			diacritics++
		}
	}
	diacriticRate := float64(diacritics) / float64(max(runes, 1))

	english := scores[0] // "en" is first
	bestLang, best := "", 0
	for i := 1; i < len(stopwords); i++ {
		if scores[i] > best {
			bestLang, best = stopwords[i].lang, scores[i]
		}
	}

	switch {
	case best == 0 && diacriticRate < 0.02:
	case bestLang != "" && best >= max(2, english+2):
		return bestLang
	case diacriticRate >= 0.04 && bestLang != "" && best >= english:
		return bestLang
	}
	if english > 0 {
		return "en"
	}
	return ""
}

// letterWords splits text into runs of letters, dropping digits and punctuation.
func letterWords(text string) []string {
	var (
		words []string
		word  strings.Builder
	)
	for _, r := range text {
		if unicode.IsLetter(r) {
			word.WriteRune(r)
			continue
		}
		if word.Len() > 0 {
			words = append(words, word.String())
			word.Reset()
		}
	}
	if word.Len() > 0 {
		words = append(words, word.String())
	}
	return words
}

func round4(value float64) float64 { return math.Round(value*1e4) / 1e4 }
