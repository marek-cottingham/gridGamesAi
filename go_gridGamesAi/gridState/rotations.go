package gridState

type RotMap map[string][6][6]GridIndex

type GridIndex struct {
	X int
	Y int
}

func GENERATE_ROTATION_MAP() RotMap {
	var rotations_raw map[string][36]int = map[string][36]int{}
	// top-left clockwise
	rotations_raw["tl_c"] = [36]int{12, 6, 0, 3, 4, 5, 13, 7, 1, 9, 10, 11, 14, 8, 2, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	// top-left anti-clockwise
	rotations_raw["tl_ac"] = [36]int{2, 8, 14, 3, 4, 5, 1, 7, 13, 9, 10, 11, 0, 6, 12, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	// top-right clockwise
	rotations_raw["tr_c"] = [36]int{0, 1, 2, 15, 9, 3, 6, 7, 8, 16, 10, 4, 12, 13, 14, 17, 11, 5,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	// top-right anti-clockwise
	rotations_raw["tr_ac"] = [36]int{0, 1, 2, 5, 11, 17, 6, 7, 8, 4, 10, 16, 12, 13, 14, 3, 9, 15,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
	// bottom-left clockwise
	rotations_raw["bl_c"] = [36]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		30, 24, 18, 21, 22, 23, 31, 25, 19, 27, 28, 29, 32, 26, 20, 33, 34, 35}
	// bottom-left anti-clockwise
	rotations_raw["bl_ac"] = [36]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		20, 26, 32, 21, 22, 23, 19, 25, 31, 27, 28, 29, 18, 24, 30, 33, 34, 35}
	// bottom-right clockwise
	rotations_raw["br_c"] = [36]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 33, 27, 21, 24, 25, 26, 34, 28, 22, 30, 31, 32, 35, 29, 23}
	// bottom-right anti-clockwise
	rotations_raw["br_ac"] = [36]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 23, 29, 35, 24, 25, 26, 22, 28, 34, 30, 31, 32, 21, 27, 33}

	var rotations RotMap = RotMap{}
	for k, v := range rotations_raw {
		rotG := [6][6]GridIndex{}
		for i := 0; i < 36; i++ {
			rotG[i/6][i%6] = GridIndex{v[i] / 6, v[i] % 6}
		}
		rotations[k] = rotG
	}
	return rotations
}
