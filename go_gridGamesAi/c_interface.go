package main

/*
#include <stdlib.h>
typedef float (*Score) (int*);
extern int* C_Minimax_Move(int* grid, int grid_size, int depth, Score score);
extern void free_arr_int(int* p);
*/
import "C"
import (
	"marek/goPentago/go_gridGamesAi/gameState.go"
	"unsafe"
)

func arr_int_c_to_go(arr_c *C.int, size int) []int {
	var slice []int = make([]int, 0, size)
	for _, c_int := range unsafe.Slice(arr_c, size) {
		slice = append(slice, int(c_int))
	}
	return slice
}

func arr_int_go_to_c(new_grid [74]int) *C.int {
	c_arr := C.malloc(C.size_t(74) * C.size_t(unsafe.Sizeof(C.int(0))))
	go_view := (*[74]C.int)(c_arr)
	for i := range new_grid {
		go_view[i] = (C.int)(new_grid[i])
	}
	return (*C.int)(c_arr)
}

func Minimax_Move(grid []int, depth int) [74]int {
	var arr [74]int
	copy(arr[:], grid)
	gs := gameState.GameStateFromArray(arr)
	gs_next := *gs.GetNext()
	return gs_next[0].AsArray()
}

//export C_Minimax_Move
func C_Minimax_Move(grid *C.int, grid_size C.int, depth C.int, score C.Score) *C.int {
	r := goCallback(score, grid)
	grid_go := arr_int_c_to_go(grid, int(grid_size))
	new_grid := Minimax_Move(grid_go, int(depth))
	c_arr := arr_int_go_to_c(new_grid)
	return c_arr
}

//export free_arr_int
func free_arr_int(p *C.int) {
	C.free(unsafe.Pointer(p))
}

func main() {}

// go build -buildmode=c-shared -o _goPentago.so
