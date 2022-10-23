package main

/*
#include <stdlib.h>
typedef float (*Score) (int*);
extern int* C_Minimax_Move(int* grid, int grid_size, int depth, Score score);
extern int* Go_Self_Play(int* grid, int grid_size, int depth, Score score);
extern void free_arr_int(int* p);
*/
import "C"
import (
	"marek/goPentago/go_gridGamesAi/gameState"
	"unsafe"
)

type score_func func(grid_arr [74]int) float32

func arr_int_c_to_go(arr_c *C.int, size int) []int {
	var slice []int = make([]int, 0, size)
	for _, c_int := range unsafe.Slice(arr_c, size) {
		slice = append(slice, int(c_int))
	}
	return slice
}

func arr_74_int_go_to_c(new_grid [74]int) *C.int {
	c_arr := C.malloc(C.size_t(74) * C.size_t(unsafe.Sizeof(C.int(0))))
	go_view := (*[74]C.int)(c_arr)
	for i := range new_grid {
		go_view[i] = (C.int)(new_grid[i])
	}
	return (*C.int)(c_arr)
}

func arr_arr_74_int_go_to_c(arr [][74]int) *C.int {
	size := len(arr)*74 + 1
	c_arr := C.malloc(C.size_t(size) * C.size_t(unsafe.Sizeof(C.int(0))))
	go_view := (*[1<<30 - 1]C.int)(c_arr)
	go_view[0] = C.int(size)
	for i, this := range arr {
		for j := range this {
			go_view[i*74+j+1] = (C.int)(this[j])
		}
	}
	return (*C.int)(c_arr)
}

func Minimax(gs *gameState.GameState, depth int, score score_func) float32 {
	if gs.IsWin() || gs.IsDraw() || depth == 0 {
		if gs.Turn.Last_player_to_move == 1 {
			return -score(gs.AsArray())
		}
		return score(gs.AsArray())
	}
	var best_score float32 = -2
	gs_next := *gs.GetNext()
	for _, this := range gs_next {
		score_this := Minimax(this, depth-1, score)
		if score_this > best_score {
			best_score = score_this
		}
	}
	if gs.Turn.Current_turn_step == 0 {
		return -best_score
	} else {
		return best_score
	}
}

func Minimax_Move(gs *gameState.GameState, depth int, score score_func) *gameState.GameState {
	gs_next := *gs.GetNext()
	var best *gameState.GameState
	var best_score float32 = -2
	for _, this := range gs_next {
		this_score := Minimax(this, depth, score)
		if this_score > best_score {
			best = this
			best_score = this_score
		}
	}
	return best
}

//export Go_Self_Play
func Go_Self_Play(grid *C.int, grid_size C.int, depth C.int, score C.Score) *C.int {
	go_score, gs := _parse(score, grid, grid_size)
	var moves [][74]int = [][74]int{gs.AsArray()}
	for !gs.IsWin() && !gs.IsDraw() {
		gs = Minimax_Move(gs, int(depth), go_score)
		moves = append(moves, gs.AsArray())
	}
	// c_arr := arr_74_int_go_to_c(gs.AsArray())
	// return c_arr
	c_arr := arr_arr_74_int_go_to_c(moves)
	return c_arr
}

//export C_Minimax_Move
func C_Minimax_Move(grid *C.int, grid_size C.int, depth C.int, score C.Score) *C.int {
	go_score, gs := _parse(score, grid, grid_size)
	new_gs := Minimax_Move(gs, int(depth), go_score)
	c_arr := arr_74_int_go_to_c(new_gs.AsArray())
	return c_arr
}

func _parse(score C.Score, grid *C.int, grid_size C.int) (score_func, *gameState.GameState) {
	go_score := go_score_factory(score)
	grid_go := arr_int_c_to_go(grid, int(grid_size))
	var arr [74]int
	copy(arr[:], grid_go)
	gs := gameState.GameStateFromArray(arr)
	return go_score, gs
}

func go_score_factory(score C.Score) score_func {
	go_score := func(grid_arr [74]int) float32 {
		c_arr := arr_74_int_go_to_c(grid_arr)
		res := goCallback(score, c_arr)
		return res
	}
	return go_score
}

//export free_arr_int
func free_arr_int(p *C.int) {
	C.free(unsafe.Pointer(p))
}

func main() {}

// go build -buildmode=c-shared -o _goPentago.so
