package gridState

type WinList [][5]GridIndex

func GENERATE_WIN_LIST() WinList {
	var win_map WinList = WinList{}
	for i := 0; i < 6; i++ {
		horz_1 := [5]GridIndex{}
		horz_2 := [5]GridIndex{}
		vert_1 := [5]GridIndex{}
		vert_2 := [5]GridIndex{}
		for j := 0; j < 5; j++ {
			horz_1[j] = GridIndex{i, j}
			horz_2[j] = GridIndex{i, j + 1}
			vert_1[j] = GridIndex{j, i}
			vert_2[j] = GridIndex{j + 1, i}
		}
		win_map = append(win_map, horz_1)
		win_map = append(win_map, horz_2)
		win_map = append(win_map, vert_1)
		win_map = append(win_map, vert_2)
	}
	diag_1 := [5]GridIndex{}
	diag_2 := [5]GridIndex{}
	diag_3 := [5]GridIndex{}
	diag_4 := [5]GridIndex{}
	for i := 0; i < 5; i++ {
		diag_1[i] = GridIndex{i, i}
		diag_2[i] = GridIndex{i, i + 1}
		diag_3[i] = GridIndex{i + 1, i}
		diag_4[i] = GridIndex{i + 1, i + 1}
	}
	win_map = append(win_map, diag_1)
	win_map = append(win_map, diag_2)
	win_map = append(win_map, diag_3)
	win_map = append(win_map, diag_4)

	return win_map
}
