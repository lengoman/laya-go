package laya

import "testing"

func TestVersion(t *testing.T) {
	if Version != "0.3.26" {
		t.Errorf("Version = %q, want 0.3.26", Version)
	}
	if LayaTargetVersion != "0.3.26" {
		t.Errorf("LayaTargetVersion = %q, want 0.3.26", LayaTargetVersion)
	}
}
