package laya

import (
	"strings"
	"testing"
)

func TestCleanEmailBody(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		contains string
		excludes string
	}{
		{
			name: "English quote header and signature",
			input: `Hi team, please help with invoice #123.

Thanks & regards,
John Doe

On Oct 10, 2024, at 10:00 AM, Support <support@example.com> wrote:
> Previous message here`,
			contains: "Hi team, please help with invoice #123.",
			excludes: "Previous message here",
		},
		{
			name: "Portuguese quote header and signature",
			input: `Olá equipe, preciso de ajuda com o boleto.

Atenciosamente,
Carlos

Em 10 de out de 2024, Suporte <suporte@empresa.com> escreveu:
> Mensagem anterior`,
			contains: "Olá equipe, preciso de ajuda com o boleto.",
			excludes: "Mensagem anterior",
		},
		{
			name: "Spanish quote and disclaimer",
			input: `Hola, necesito soporte urgente con el sistema.

Saludos cordiales,
Maria

Este mensaje es confidencial y para uso exclusivo del destinatario.
El 10 de octubre de 2024, Soporte <soporte@ejemplo.com> escribió:
> Texto previo`,
			contains: "Hola, necesito soporte urgente con el sistema.",
			excludes: "Texto previo",
		},
		{
			name: "Device footer",
			input: `Please reset my password.

Sent from my iPhone`,
			contains: "Please reset my password.",
			excludes: "Sent from my iPhone",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CleanEmailBody(tc.input, 3000)
			if !strings.Contains(got, tc.contains) {
				t.Errorf("CleanEmailBody output %q does not contain %q", got, tc.contains)
			}
			if tc.excludes != "" && strings.Contains(got, tc.excludes) {
				t.Errorf("CleanEmailBody output %q contains excluded string %q", got, tc.excludes)
			}
		})
	}
}

func TestEmailState(t *testing.T) {
	state := EmailState(
		"Invoice question",
		"Please check my invoice.\n\nSent from my iPhone",
		"user@example.com",
		true,
		3000,
		map[string]any{"priority": "high"},
	)

	if state["subject"] != "Invoice question" {
		t.Errorf("subject = %v, want Invoice question", state["subject"])
	}
	if state["from"] != "user@example.com" {
		t.Errorf("from = %v, want user@example.com", state["from"])
	}
	if state["priority"] != "high" {
		t.Errorf("priority = %v, want high", state["priority"])
	}
	body, ok := state["body"].(string)
	if !ok || strings.Contains(body, "Sent from my iPhone") {
		t.Errorf("body = %v, should have cleaned device footer", state["body"])
	}
}
