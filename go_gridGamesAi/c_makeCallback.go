package main

/*
#include <stdlib.h>
#include <stdio.h>
typedef float (*Score) (int*);
float makeMyCallback(Score f, int* arr_grid){
	return f(arr_grid);
};
*/
import "C"

func goCallback(score C.Score, grid *C.int) float32 {
	r := C.makeMyCallback(score, grid)
	return float32(r)
}
