package gridState_test

import (
	"marek/goPentago/go_gridGamesAi/gridState"
	"testing"
)

func TestInitialiseGrid(t *testing.T) {
	g := gridState.NewTwoPlayerGridState()
	if g.Grid_0 == nil {
		t.Error("grid_0 is nil")
	}
	if g.Grid_0[0][0] != 0 {
		t.Error("grid_0[0][0] is not 0")
	}
	if g.Grid_0[5][5] != 0 {
		t.Error("grid_0[5][5] is not 0")
	}
}

func TestPlaceOnSingleGrid(t *testing.T) {
	g := gridState.NewSingleGrid()
	g2 := g.PlaceOn(2, 3)
	if g[2][3] != 0 {
		t.Error("g[2][3] is not 0")
	}
	if g2[2][3] != 1 {
		t.Error("grid[2][3] is not 1")
	}
}

func TestPlaceOnTwoPlayerGridState(t *testing.T) {
	g := gridState.NewTwoPlayerGridState()
	g2 := g.PlaceOn(0, 2, 3)
	if g.Grid_0[2][3] != 0 {
		t.Error("grid_0[2][3] is not 0")
	}
	if g2.Grid_0[2][3] != 1 {
		t.Error("grid_0[2][3] is not 1")
	}
	g3 := g2.PlaceOn(1, 2, 4)
	if g2.Grid_1[2][4] != 0 {
		t.Error("grid_1[2][4] is not 0")
	}
	if g3.Grid_1[2][4] != 1 {
		t.Error("grid_1[2][4] is not 1")
	}

}

func TestVaildPlacementsGridState(t *testing.T) {
	g := gridState.NewTwoPlayerGridState().PlaceOn(0, 2, 3).PlaceOn(1, 2, 4)
	next_g := g.GetValidPlacements(0)
	if len(next_g) != 34 {
		t.Error("len(next_g) is not 34")
	}
}

func TestRotateGridState(t *testing.T) {
	g := gridState.NewTwoPlayerGridState().PlaceOn(0, 2, 2)
	g2 := g.Rotate("tl_c")
	if g2.Grid_0[2][2] != 0 {
		t.Error("grid_0[2][2] is not 0")
	}
	if g2.Grid_0[2][0] != 1 {
		t.Error("grid_0[2][0] is not 1")
	}
	if g.Grid_0[2][2] != 1 {
		t.Error("grid_0[2][2] is not 1")
	}
}

func TestWinSingleGrid(t *testing.T) {
	g := gridState.NewSingleGrid().PlaceOn(0, 1).PlaceOn(1, 1).PlaceOn(
		2, 1).PlaceOn(3, 1).PlaceOn(4, 1)
	if g.IsWin() != true {
		t.Error("g.IsWin() is not true")
	}
	if len(gridState.WinsList) != 28 {
		t.Errorf("len(WinsList) is %v not 28", len(gridState.WinsList))
	}
}
