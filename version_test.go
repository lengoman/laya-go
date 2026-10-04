package laya

import "testing"

func TestVersion(t *testing.T) {
	if Version != "0.2.0" {
		t.Errorf("Version = %q, want 0.2.0", Version)
	}
	if LayaTargetVersion != "0.3.26" {
		t.Errorf("LayaTargetVersion = %q, want 0.3.26", LayaTargetVersion)
	}
}
