package gameState_test

import (
	"marek/goPentago/go_gridGamesAi/gameState"
	"testing"
)

func TestGameStateInit(t *testing.T) {
	gs := gameState.NewGameState()
	if gs.Grid == nil {
		t.Error("Grid is nil")
	}
	if gs.Turn == nil {
		t.Error("Turn is nil")
	}
}

func TestGameStateWinInfo(t *testing.T) {
	gs := gameState.NewGameState()
	gs, _ = gs.Place(0, 0)
	win := gs.IsWin()
	if win {
		t.Error("GameState.IsWin() returned true")
	}
	draw := gs.IsDraw()
	if draw {
		t.Error("GameState.IsDraw() returned true")
	}
	winPlayer := gs.GetWinPlayer()
	if winPlayer != -1 {
		t.Error("GameState.WinPlayer() returned ", winPlayer)
	}
}

func TestGameStateNext(t *testing.T) {
	gs := gameState.NewGameState()
	gs, _ = gs.Place(0, 0)
	next := gs.GetNext()
	if len(*next) != 8 {
		t.Error("GameState.GetNext() returned ", len(*next), " states")
	}
	gs2, _ := gs.Rotate("br_ac")
	next2 := gs2.GetNext()
	if len(*next2) != 35 {
		t.Error("GameState.GetNext() returned ", len(*next2), " states")
	}
}

func TestToFromArrayGameState(t *testing.T) {
	gs, _ := gameState.NewGameState().Place(0, 0)
	gs, _ = gs.Rotate("br_ac")
	gs, _ = gs.Place(1, 1)
	arr := gs.AsArray()
	gs2 := gameState.GameStateFromArray(arr)
	for i := range gs.Grid.Grid_0 {
		for j := range gs.Grid.Grid_0[i] {
			if gs.Grid.Grid_0[i][j] != gs2.Grid.Grid_0[i][j] {
				t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Grid_0")
			}
			if gs.Grid.Grid_1[i][j] != gs2.Grid.Grid_1[i][j] {
				t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Grid_1")
			}
		}
	}
	if gs.Turn.Last_player_to_move != gs2.Turn.Last_player_to_move {
		t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Last_player_to_move")
	}
	if gs.Turn.Current_player != gs2.Turn.Current_player {
		t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Current_player")
	}
	if gs.Turn.Current_turn_step != gs2.Turn.Current_turn_step {
		t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Current_turn_step")
	}
	if gs.Turn.Total_moves != gs2.Turn.Total_moves {
		t.Error("GameStateFromArray(GameState.AsArray()) did not return the same Total_moves")
	}
}
