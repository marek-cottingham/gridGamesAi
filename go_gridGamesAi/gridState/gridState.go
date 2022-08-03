package gridState

var RotationMap = GENERATE_ROTATION_MAP()

var WinsList = GENERATE_WIN_LIST()

type SingleGrid [6][6]int

func NewSingleGrid() *SingleGrid {
	var grid SingleGrid
	for i := range grid {
		grid[i] = [6]int{}
	}
	return &grid
}

func (g *SingleGrid) Copy() *SingleGrid {
	var grid SingleGrid
	for i := range grid {
		for j := range grid[i] {
			grid[i][j] = g[i][j]
		}
	}
	return &grid
}

func (g *SingleGrid) PlaceOn(row int, col int) *SingleGrid {
	grid := g.Copy()
	grid[row][col] = 1
	return grid
}

func (g *SingleGrid) Rotate(rotation string) *SingleGrid {
	grid := NewSingleGrid()
	for i := range grid {
		for j := range grid[i] {
			take_index := RotationMap[rotation][i][j]
			grid[i][j] = g[take_index.X][take_index.Y]
		}
	}
	return grid
}

func (g *SingleGrid) IsWin() bool {
	for _, win := range WinsList {
		if g[win[0].X][win[0].Y] == 1 &&
			g[win[1].X][win[1].Y] == 1 &&
			g[win[2].X][win[2].Y] == 1 &&
			g[win[3].X][win[3].Y] == 1 &&
			g[win[4].X][win[4].Y] == 1 {
			return true
		}
	}
	return false
}

type TwoPlayerGridState struct {
	Grid_0 *SingleGrid
	Grid_1 *SingleGrid
}

func NewTwoPlayerGridState() *TwoPlayerGridState {
	return &TwoPlayerGridState{
		Grid_0: NewSingleGrid(),
		Grid_1: NewSingleGrid(),
	}
}

func (g *TwoPlayerGridState) PlaceOn(player int, row int, col int) *TwoPlayerGridState {
	grid := *g
	if player == 0 {
		grid.Grid_0 = grid.Grid_0.PlaceOn(row, col)
	}
	if player == 1 {
		grid.Grid_1 = grid.Grid_1.PlaceOn(row, col)
	}
	return &grid
}

func (g *TwoPlayerGridState) CombinedGrid() *SingleGrid {
	grid := NewSingleGrid()
	for i := range grid {
		for j := range grid[i] {
			grid[i][j] = g.Grid_0[i][j] + g.Grid_1[i][j]
		}
	}
	return grid
}

func (g *TwoPlayerGridState) GetValidPlacements(player int) []*TwoPlayerGridState {
	grid := g.CombinedGrid()
	validPlacements := make([]*TwoPlayerGridState, 0, 36)
	for i := range grid {
		for j := range grid[i] {
			if grid[i][j] == 0 {
				validPlacements = append(validPlacements, g.PlaceOn(player, i, j))
			}
		}
	}
	return validPlacements
}

func (g *TwoPlayerGridState) Rotate(rotation string) *TwoPlayerGridState {
	grid := *g
	grid.Grid_0 = grid.Grid_0.Rotate(rotation)
	grid.Grid_1 = grid.Grid_1.Rotate(rotation)
	return &grid
}
