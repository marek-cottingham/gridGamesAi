package turnTracker_test

import (
	turn "marek/goPentago/go_gridGamesAi/turnTracker"
	"testing"
)

func TestNewTurnTracker(t *testing.T) {
	tt := turn.NewTurnTracker(2, 3)
	tt.GetIncremented()
	if tt.Number_players != 2 {
		t.Errorf("expected 2, got %d", tt.Number_players)
	}
	if tt.Number_turn_steps != 3 {
		t.Errorf("expected 3, got %d", tt.Number_turn_steps)
	}
	if tt.Current_player != 0 {
		t.Errorf("expected 0, got %d", tt.Current_player)
	}
	if tt.Current_turn_step != 0 {
		t.Errorf("expected 0, got %d", tt.Current_turn_step)
	}
	if tt.Total_moves != 0 {
		t.Errorf("expected 0, got %d", tt.Total_moves)
	}
	if tt.Last_player_to_move != -1 {
		t.Errorf("expected -1, got %d", tt.Last_player_to_move)
	}
}

func TestIncrementTurnTracker(t *testing.T) {
	tt := turn.NewTurnTracker(2, 3)
	tt = tt.GetIncremented()
	if tt.Current_player != 0 {
		t.Errorf("expected 1, got %d", tt.Current_player)
	}
	if tt.Current_turn_step != 1 {
		t.Errorf("expected 1, got %d", tt.Current_turn_step)
	}
	if tt.Total_moves != 1 {
		t.Errorf("expected 1, got %d", tt.Total_moves)
	}
	if tt.Last_player_to_move != 0 {
		t.Errorf("expected 0, got %d", tt.Last_player_to_move)
	}
	tt = tt.GetIncremented().GetIncremented()
	if tt.Current_player != 1 {
		t.Errorf("expected 1, got %d", tt.Current_player)
	}
	if tt.Current_turn_step != 0 {
		t.Errorf("expected 0, got %d", tt.Current_turn_step)
	}
	if tt.Total_moves != 3 {
		t.Errorf("expected 3, got %d", tt.Total_moves)
	}
	tt = tt.GetIncremented()
	if tt.Last_player_to_move != 1 {
		t.Errorf("expected 1, got %d", tt.Last_player_to_move)
	}
}
