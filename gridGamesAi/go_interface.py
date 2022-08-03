import ctypes
import numpy as np
from .pentago.gameState import PentagoGameState

c_int = ctypes.c_int
c_int_p = ctypes.POINTER(c_int)

lib = ctypes.cdll.LoadLibrary('./go_gridgamesAi/_goPentago.so')
lib.C_Minimax_Move.argtypes = [c_int_p, c_int, c_int]
lib.C_Minimax_Move.restype = c_int_p
lib.free_arr_int.argtypes = [c_int_p]

def goMinimaxMove(gameState: PentagoGameState, depth: int, score_gameState: callable):

    @ctypes.CFUNCTYPE(c_int, c_int_p)
    def score(c_arr: c_int_p):
        arr = _decode_c_arr_74_int(c_arr)
        return score_gameState(PentagoGameState.fromNumpy(arr))

    c_arr, c_arr_size = _ndarrInt_to_CintArray(gameState.asNumpy())
    out_c_arr = lib.C_Minimax_Move(c_arr, c_arr_size, c_int(depth))
    decoded = _decode_c_arr_74_int(out_c_arr)
    lib.free_arr_int(out_c_arr)
    return PentagoGameState.fromNumpy(decoded)

def _decode_c_arr_74_int(out_c_arr):
    decoded = np.fromiter(out_c_arr, dtype=np.int32, count=74)
    return decoded

def _ndarrInt_to_CintArray(ndarr: np.ndarray):
    data = ndarr.astype(np.int32)
    data_p = data.ctypes.data_as(c_int_p)
    return data_p, c_int(len(data))