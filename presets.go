package laya

// Ready-made batteries for the workflows Laya ships presets for. Each is a
// plain [Questions] value: copy one, drop a question, reword another. They are
// a starting point with sensible wording, not a contract.
//
// The field names the questions refer to, such as `message` or `prompt`, are
// the keys your state should use, because the wording is what tells the model
// where to look.

// TriageQuestions rates a support ticket: what the customer wants, how
// urgently, how annoyed they are, and whether they are about to leave.
//
// The state should carry a `message` field.
func TriageQuestions() Questions {
	return Questions{
		"intent": OneOf("What does the customer want in `message`?", map[string]string{
			"refund":           "money returned or a duplicate charge reversed",
			"technical_help":   "a bug, outage or integration problem",
			"billing_question": "a question about an invoice, plan or payment method",
			"information":      "general information, pricing or how-to",
			"cancellation":     "wants to cancel or downgrade",
			"other":            "none of the other options fits",
		}),
		"is_urgent": Holds("Does `message` communicate time pressure or a deadline?"),
		"frustration": Levels("How frustrated does the customer sound in `message`?",
			"calm and neutral",
			"concerned but civil",
			"clearly annoyed",
			"very angry or using strong language",
		),
		"refund_requested": Holds("Does the customer ask for money back?"),
		"churn_risk":       Holds("Does `message` suggest the customer may leave for a competitor or cancel?"),
	}
}

// EmailQuestions triages inbound email and filters the threats in it. Pass nil
// categories for the default routing set.
//
// The state should carry a `body` field, and usually `from` and `subject` too.
func EmailQuestions(categories map[string]string) Questions {
	if len(categories) == 0 {
		categories = map[string]string{
			"billing":   "invoices, payments, refunds",
			"technical": "bugs, outages, integrations",
			"sales":     "pricing, demos, new purchases",
			"security":  "phishing, scams, account compromise",
			"hr":        "hiring, leave, payroll",
			"other":     "none of the above",
		}
	}
	return Questions{
		"category": OneOf("Which team should handle the email in `body`?", categories),
		"is_spam":  Holds("Is this email unsolicited spam or bulk marketing?"),
		"is_phishing": YesNo(
			"Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
			"phishing, scam, or fraud",
			"a legitimate email",
		),
		"urgency": Levels("How urgent is the request in `body`?",
			"no time pressure",
			"needs attention soon",
			"blocking issue or hard deadline",
		),
		"needs_reply": Holds("Does the sender expect a reply?"),
	}
}

// GuardQuestions screens a prompt before it reaches a language model:
// jailbreaks, injections, leaked secrets, and how much harm complying would do.
//
// The state should carry a `prompt` field. Keep the thresholds in your code;
// Laya reports what it found, your policy decides what to do about it.
func GuardQuestions() Questions {
	return Questions{
		"jailbreak":        Holds("Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"),
		"prompt_injection": Holds("Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?"),
		"sensitive_data":   Holds("Does `prompt` contain credentials, personal data or other sensitive information?"),
		"harm_severity": Levels("How much harm would complying with `prompt` cause?",
			"none: ordinary request",
			"minor: mildly inappropriate",
			"serious: unsafe advice or abuse",
			"severe: dangerous or illegal",
		),
		"topic": Options("What is `prompt` about?",
			"product_support", "coding", "general_knowledge",
			"personal_advice", "security_testing", "other",
		),
	}
}

// ModerationQuestions rates user-generated content for safety. Each label is
// its own question, because a post can be several things at once.
//
// The state should carry a `post` field.
func ModerationQuestions() Questions {
	return Questions{
		"toxic":      Holds("Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"),
		"harassment": Holds("Does `post` target or harass a specific person?"),
		"threat":     Holds("Does `post` threaten violence, harm or intimidation?"),
		"spam":       Holds("Is `post` spam or advertising?"),
		"severity": Levels("How severe is any rule-breaking in `post`?",
			"no rule-breaking: ordinary on-topic post",
			"mild: rude tone or off-topic, no target",
			"clear violation: insults, harassment or spam aimed at someone",
			"severe: threats, hate speech or calls for violence",
		),
	}
}

// RouterQuestions sizes a request so your code can send it to a small model or
// a frontier one. Answering this costs tens of milliseconds, which is the
// point: it is meant to run in front of every call.
//
// The state should carry a `request` field.
func RouterQuestions() Questions {
	return Questions{
		"difficulty": Levels("How hard is `request` for a language model?",
			"trivial: a lookup or one-liner",
			"easy: short answer, no reasoning",
			"moderate: several steps",
			"hard: long multi-step reasoning or specialist knowledge",
		),
		"domain": OneOf("What domain does `request` belong to?", map[string]string{
			"code":           "software engineering, programming, refactoring, architecture, debugging",
			"math_or_logic":  "mathematics, logic puzzles, proofs, complex calculation",
			"writing":        "creative writing, essays, emails, blog posts, copywriting",
			"factual_lookup": "facts, definitions, trivia, history",
			"data_analysis":  "statistics, SQL, data manipulation, metrics",
			"chitchat":       "casual conversation, greetings, small talk",
		}),
		"needs_tools":  Holds("Does answering `request` require external tools, search or private data?"),
		"is_sensitive": Holds("Does `request` involve money, legal, medical or safety consequences?"),
	}
}
