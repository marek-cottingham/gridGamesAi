package gameState

import (
	"errors"
	"marek/goPentago/go_gridGamesAi/gridState"
	"marek/goPentago/go_gridGamesAi/turnTracker"
)

type WinInfo struct {
	IsWin     bool
	IsDraw    bool
	WinPlayer int
}

type GameState struct {
	Grid   *gridState.TwoPlayerGridState
	Turn   *turnTracker.TurnTracker
	WinInf *WinInfo
	Next   *[]*GameState
}

func NewGameState() *GameState {
	return &GameState{
		Grid: gridState.NewTwoPlayerGridState(),
		Turn: turnTracker.NewTurnTracker(2, 2),
	}
}

func (gs *GameState) Place(x int, y int) (*GameState, error) {
	if gs.Grid.CombinedGrid()[x][y] != 0 {
		return nil, errors.New("cannot place in occupied cell")
	}
	next_grid := gs.Grid.PlaceOn(gs.Turn.Current_player, x, y)
	next_turn := gs.Turn.GetIncremented()
	return &GameState{next_grid, next_turn, nil, nil}, nil
}

func (gs *GameState) Rotate(rotation string) (*GameState, error) {
	next_grid := gs.Grid.Rotate(rotation)
	next_turn := gs.Turn.GetIncremented()
	return &GameState{next_grid, next_turn, nil, nil}, nil
}

func (gs *GameState) PopulateWinInfo() {
	win_player_0 := gs.Grid.Grid_0.IsWin()
	win_player_1 := gs.Grid.Grid_1.IsWin()
	if win_player_0 && win_player_1 {
		gs.WinInf = &WinInfo{true, false, (gs.Turn.Last_player_to_move + 1) % 2}
		return
	}
	if win_player_0 || win_player_1 {
		gs.WinInf = &WinInfo{true, false, gs.Turn.Last_player_to_move}
		return
	}
	comb_grid := gs.Grid.CombinedGrid()
	for i := range comb_grid {
		for j := range comb_grid[i] {
			if comb_grid[i][j] == 0 {
				gs.WinInf = &WinInfo{false, false, -1}
				return
			}
		}
	}
	gs.WinInf = &WinInfo{false, true, -1}
}

func (gs *GameState) IsWin() bool {
	if gs.WinInf == nil {
		gs.PopulateWinInfo()
	}
	return gs.WinInf.IsWin
}

func (gs *GameState) IsDraw() bool {
	if gs.WinInf == nil {
		gs.PopulateWinInfo()
	}
	return gs.WinInf.IsDraw
}

func (gs *GameState) GetWinPlayer() int {
	if gs.WinInf == nil {
		gs.PopulateWinInfo()
	}
	return gs.WinInf.WinPlayer
}

func (gs *GameState) GetNext() *[]*GameState {
	if gs.Next == nil {
		gs.PopulateNext()
	}
	return gs.Next
}

func (gs *GameState) PopulateNext() {
	var next []*GameState
	if gs.Turn.Current_turn_step == 0 {
		next = make([]*GameState, 0, 36)
		valid_grids := gs.Grid.GetValidPlacements(gs.Turn.Current_player)
		for _, valid_grid := range valid_grids {
			next = append(next, &GameState{valid_grid, gs.Turn.GetIncremented(), nil, nil})
		}
	} else {
		next = make([]*GameState, 0, 8)
		for rot_key := range gridState.RotationMap {
			next = append(next, &GameState{gs.Grid.Rotate(rot_key), gs.Turn.GetIncremented(), nil, nil})
		}
	}
	gs.Next = &next
}

func (gs *GameState) AsArray() [74]int {
	array := [74]int{}
	array[0] = gs.Turn.Current_player
	array[1] = gs.Turn.Current_turn_step
	for i := range gs.Grid.Grid_0 {
		for j := range gs.Grid.Grid_0[i] {
			array[i*6+j+2] = gs.Grid.Grid_0[i][j]
		}
	}
	for i := range gs.Grid.Grid_1 {
		for j := range gs.Grid.Grid_1[i] {
			array[i*6+j+38] = gs.Grid.Grid_1[i][j]
		}
	}
	return array
}

func GameStateFromArray(array [74]int) *GameState {
	grid_0 := gridState.SingleGrid{}
	grid_1 := gridState.SingleGrid{}
	for i := range grid_0 {
		for j := range grid_0[i] {
			grid_0[i][j] = array[i*6+j+2]
		}
	}
	for i := range grid_1 {
		for j := range grid_1[i] {
			grid_1[i][j] = array[i*6+j+38]
		}
	}
	grid := gridState.TwoPlayerGridState{Grid_0: &grid_0, Grid_1: &grid_1}

	var total_moves int
	var comb_grid gridState.SingleGrid = *grid.CombinedGrid()
	for i := range comb_grid {
		for j := range comb_grid[i] {
			total_moves += comb_grid[i][j] * 2
		}
	}

	var previous_player int
	if array[1] == 1 {
		previous_player = array[0]
		total_moves -= 1
	} else {
		previous_player = (array[0] + 1) % 2
	}

	turn := turnTracker.TurnTracker{
		Number_players:      2,
		Number_turn_steps:   2,
		Current_player:      array[0],
		Current_turn_step:   array[1],
		Last_player_to_move: previous_player,
		Total_moves:         total_moves,
	}
	return &GameState{
		Grid: &grid,
		Turn: &turn,
	}
}
