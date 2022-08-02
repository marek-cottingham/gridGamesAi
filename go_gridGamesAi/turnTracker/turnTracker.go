package turnTracker

type TurnTracker struct {
	Number_players      int
	Number_turn_steps   int
	Current_player      int
	Current_turn_step   int
	Total_moves         int
	Last_player_to_move int
}

func NewTurnTracker(number_players int, number_turn_steps int) *TurnTracker {
	return &TurnTracker{
		Number_players:      number_players,
		Number_turn_steps:   number_turn_steps,
		Current_player:      0,
		Current_turn_step:   0,
		Total_moves:         0,
		Last_player_to_move: -1,
	}
}

func (tt *TurnTracker) GetNextPlayer() int {
	is_last_turn_step := tt.Current_turn_step == tt.Number_turn_steps-1
	if is_last_turn_step {
		return (tt.Current_player + 1) % tt.Number_players
	} else {
		return tt.Current_player
	}
}

func (tt *TurnTracker) GetNextTurnStep() int {
	return (tt.Current_turn_step + 1) % tt.Number_turn_steps
}

func (tt *TurnTracker) GetIncremented() *TurnTracker {
	return &TurnTracker{
		tt.Number_players, tt.Number_turn_steps, tt.GetNextPlayer(),
		tt.GetNextTurnStep(), tt.Total_moves + 1, tt.Current_player,
	}
}
