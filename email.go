package laya

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Email cleaning and structuring utilities matching laya.email.
//
// These markers cover English, Portuguese, Spanish and French mail clients,
// removing quoted email history, signatures and disclaimers.

var (
	quoteHeaders = []*regexp.Regexp{
		regexp.MustCompile(`(?i)^\s*On .{0,300}wrote:\s*$`),
		regexp.MustCompile(`(?i)^\s*Em [^\n]{0,300}\d[^\n]{0,300}escreveu:\s*$`),
		regexp.MustCompile(`(?i)^\s*El [^\n]{0,300}\d[^\n]{0,300}escribi[óo]:\s*$`),
		regexp.MustCompile(`(?i)^\s*Le [^\n]{0,300}\d[^\n]{0,300}a [eé]crit\s*:\s*$`),
		regexp.MustCompile(`(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}`),
		regexp.MustCompile(`(?i)^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado)|Message d'origine)\s*-{2,}`),
		regexp.MustCompile(`^\s*_{8,}\s*$`),
		regexp.MustCompile(`(?i)^\s*From:\s.*[@<]`),
		regexp.MustCompile(`(?i)^\s*De\s*:\s.*[@<]`),
	}

	attributionTail = regexp.MustCompile(`(?i)^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]|a [eé]crit)\s*:\s*$`)
	attributionHead = regexp.MustCompile(`(?i)^\s*(On|Em|El|Le)\s+.*\d`)

	headerFromName = regexp.MustCompile(`(?i)^\s*(De|From)\s*:\s+\S`)
	headerNext     = regexp.MustCompile(`(?i)^\s*(Enviad[oa]( em| el)?:\s|Envoy[ée]( le)?\s*:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4})`)

	signoffHead = regexp.MustCompile(
		`(?i)^\s*(best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)` +
			`(\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?`,
	)
	signoffTail  = regexp.MustCompile(`^[\s,;:!.]*(?:[^\W\d_][\w'-]*[\s,.]*){0,3}$`)
	signoffToken = regexp.MustCompile(`[^\W\d_][\w'-]*`)

	multilingualSignoffs = regexp.MustCompile(
		`(?i)^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?` +
			`( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|` +
			`(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|` +
			`(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|` +
			`bonne journée)[\s,!.]*$`,
	)

	devicePattern = `(?:iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows)`
	deviceFooter  = regexp.MustCompile(
		`(?i)^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?|` +
			`envoy[ée] (depuis|de) (mon |ma |mes )?)` +
			` (` + devicePattern + `)( (` + devicePattern + `|para|for|no|na|\d+|phone|device|pro|max|mini|plus|using [a-z][a-z0-9_.+-]*))*` +
			`|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$`,
	)

	disclaimerRe = regexp.MustCompile(
		`(?i)(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|disclos|unauthori[sz]ed)|` +
			`\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|` +
			`if you (have )?received this (e-?mail|message) in error|` +
			`\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|` +
			`\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}(destinatári|destinatari|pessoa|persona|entidade|entidad)|` +
			`\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|` +
			`\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|` +
			`\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|` +
			`\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir|` +
			`\b(ce|cet|cette) (message|e-?mail|mail|courriel)\b[^.]{0,80}(confidentiel|privil[eé]gi)|` +
			`\bavez re[çc]u (ce|cet|cette) (message|e-?mail|mail)\b[^.]{0,20} par erreur|` +
			`\b(usage exclusif|exclusivement|uniquement)\b[^.]{0,30}destinataire)`,
	)

	multiSpaceRe = regexp.MustCompile(`[ \t]+`)
	paraSplitRe  = regexp.MustCompile(`\n\s*\n`)
)

func isEnglishSignoff(line string) bool {
	loc := signoffHead.FindStringIndex(line)
	if loc == nil {
		return false
	}
	tail := dropCombiningMarks(line[loc[1]:])
	if !signoffTail.MatchString(tail) {
		return false
	}
	tokens := signoffToken.FindAllString(tail, -1)
	for _, token := range tokens {
		r, _ := utf8.DecodeRuneInString(token)
		if !unicode.IsUpper(r) && !unicode.IsTitle(r) {
			return false
		}
	}
	return true
}

func dropCombiningMarks(text string) string {
	var kept []rune
	for _, r := range text {
		if unicode.Is(unicode.M, r) && len(kept) > 0 && !unicode.IsSpace(kept[len(kept)-1]) {
			continue
		}
		kept = append(kept, r)
	}
	return string(kept)
}

func isSignatureMarker(line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "--" {
		return true
	}
	if isEnglishSignoff(line) {
		return true
	}
	return multilingualSignoffs.MatchString(line)
}

func startsNewSentence(line string) bool {
	for _, r := range line {
		if unicode.IsLetter(r) {
			return unicode.IsUpper(r)
		}
	}
	return false
}

func splitFusedLines(sentence string) []string {
	if !strings.Contains(sentence, "\n") {
		return []string{sentence}
	}
	var (
		pieces []string
		buf    string
	)
	for _, rawLine := range strings.Split(sentence, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		if buf != "" && startsNewSentence(line) {
			pieces = append(pieces, buf)
			buf = line
		} else {
			if buf != "" {
				buf = buf + " " + line
			} else {
				buf = line
			}
		}
	}
	if buf != "" {
		pieces = append(pieces, buf)
	}
	return pieces
}

func stripDisclaimer(paragraph string) string {
	if !disclaimerRe.MatchString(paragraph) {
		return paragraph
	}
	parts := sentenceSplit(paragraph)
	var pieces []string
	for _, p := range parts {
		if disclaimerRe.MatchString(p) {
			pieces = append(pieces, splitFusedLines(p)...)
		} else {
			pieces = append(pieces, p)
		}
	}
	var kept []string
	for _, p := range pieces {
		if !disclaimerRe.MatchString(p) {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, " ")
}

func sentenceSplit(text string) []string {
	var (
		res   []string
		start = 0
		runes = []rune(text)
	)
	for i := 0; i < len(runes); i++ {
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			if i+1 < len(runes) && unicode.IsSpace(runes[i+1]) {
				part := strings.TrimSpace(string(runes[start : i+1]))
				if part != "" {
					res = append(res, part)
				}
				for i+1 < len(runes) && unicode.IsSpace(runes[i+1]) {
					i++
				}
				start = i + 1
			}
		}
	}
	if start < len(runes) {
		tail := strings.TrimSpace(string(runes[start:]))
		if tail != "" {
			res = append(res, tail)
		}
	}
	return res
}

// CleanEmailBody removes quoted email history, signatures and disclaimers to keep input focused.
func CleanEmailBody(body string, maxChars int) string {
	if maxChars <= 0 {
		maxChars = 3000
	}
	text := strings.ReplaceAll(body, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.ReplaceAll(text, "\\n", "\n")

	if len(text) > maxChars*4 {
		text = text[:maxChars*4]
	}

	src := strings.Split(text, "\n")
	var lines []string

	for i, line := range src {
		isQuote := false
		for _, p := range quoteHeaders {
			if p.MatchString(line) && len(lines) > 0 {
				isQuote = true
				break
			}
		}
		if isQuote {
			break
		}
		if len(lines) > 0 && headerFromName.MatchString(line) && i+1 < len(src) && headerNext.MatchString(src[i+1]) {
			break
		}
		if attributionTail.MatchString(line) && len(lines) > 0 {
			if attributionHead.MatchString(lines[len(lines)-1]) {
				lines = lines[:len(lines)-1]
			}
			break
		}
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), ">") {
			continue
		}
		lines = append(lines, strings.TrimRight(line, " \t\r"))
	}

	cut := len(lines)
	startScan := max(1, min(int(float64(len(lines))*0.6), len(lines)-8))
	for i := startScan; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		n := len(trimmed)
		if (n <= 40 && isSignatureMarker(lines[i])) || (n <= 60 && deviceFooter.MatchString(lines[i])) {
			cut = i
			break
		}
	}
	lines = lines[:cut]

	rawParas := paraSplitRe.Split(strings.Join(lines, "\n"), -1)
	var paras []string
	for _, p := range rawParas {
		cleaned := stripDisclaimer(p)
		if strings.TrimSpace(cleaned) != "" {
			paras = append(paras, strings.TrimSpace(cleaned))
		}
	}

	joined := strings.Join(paras, "\n\n")
	joined = multiSpaceRe.ReplaceAllString(joined, " ")
	runes := []rune(joined)
	if len(runes) > maxChars {
		return string(runes[:maxChars])
	}
	return joined
}

// EmailState constructs a clean state dictionary for email classification.
func EmailState(subject, body, sender string, clean bool, maxChars int, extra map[string]any) map[string]any {
	if maxChars <= 0 {
		maxChars = 3000
	}
	cleanedBody := body
	if clean {
		cleanedBody = CleanEmailBody(body, maxChars)
	}
	state := map[string]any{
		"subject": strings.TrimSpace(subject),
		"body":    cleanedBody,
	}
	if sender != "" {
		state["from"] = sender
	}
	for k, v := range extra {
		if v != nil {
			state[k] = v
		}
	}
	return state
}
