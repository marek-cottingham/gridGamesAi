import ctypes
from typing import List
import numpy as np
from .pentago.gameState import PentagoGameState

c_int = ctypes.c_int
c_float = ctypes.c_float
c_int_p = ctypes.POINTER(c_int)

lib = ctypes.cdll.LoadLibrary('./go_gridgamesAi/_goPentago.so')
lib.C_Minimax_Move.argtypes = [c_int_p, c_int, c_int]
lib.C_Minimax_Move.restype = c_int_p
lib.Go_Self_Play.argtypes = [c_int_p, c_int, c_int]
lib.Go_Self_Play.restype = c_int_p
lib.free_arr_int.argtypes = [c_int_p]

exception_raised_inside_ctypes_callback = False
keyboard_interrupt_inside_ctypes_callback = False

def goMinimaxMove(gameState: PentagoGameState, depth: int, score_gameState: callable):
    score, c_arr, c_arr_size = _parseInputs(gameState, score_gameState)
    out_c_arr = lib.C_Minimax_Move(c_arr, c_arr_size, c_int(depth), score)
    decoded = _decode_c_arr_74_int(out_c_arr)
    lib.free_arr_int(out_c_arr)
    return PentagoGameState.fromNumpy(decoded)

def _parseInputs(gameState: PentagoGameState, score_gameState: callable):
    @ctypes.CFUNCTYPE(c_float, c_int_p)
    def score(c_arr: c_int_p):
        try:
            arr = _decode_c_arr_74_int(c_arr)
            return score_gameState(PentagoGameState.fromNumpy(arr))
        except Exception as e:
            global exception_raised_inside_ctypes_callback
            exception_raised_inside_ctypes_callback = True
            raise e
        except KeyboardInterrupt as e:
            global keyboard_interrupt_inside_ctypes_callback
            keyboard_interrupt_inside_ctypes_callback = True
            return 0.0
    c_arr, c_arr_size = _ndarrInt_to_CintArray(gameState.asNumpy())
    return score, c_arr, c_arr_size

def goSelfPlay(
    gameState: PentagoGameState, depth: int, score_gameState: callable
) -> List[PentagoGameState]:
    score, c_arr, c_arr_size = _parseInputs(gameState, score_gameState)
    out_c_arr = lib.Go_Self_Play(c_arr, c_arr_size, c_int(depth), score)
    _raise_if_reported_exception()
    decoded = _decode_c_arr_arr_74_int(out_c_arr)
    lib.free_arr_int(out_c_arr)
    return [PentagoGameState.fromNumpy(d) for d in decoded]

def _raise_if_reported_exception():
    if exception_raised_inside_ctypes_callback:
        raise Exception("Exception raised inside ctypes callback")
    if keyboard_interrupt_inside_ctypes_callback:
        print("")
        print("Keyboard interrupt")
        exit()

def _decode_c_arr_74_int(out_c_arr):
    decoded = np.fromiter(out_c_arr, dtype=np.int32, count=74)
    return decoded

def _decode_c_arr_arr_74_int(out_c_arr):
    size = out_c_arr[0]
    decoded = np.fromiter(out_c_arr, dtype=np.int32, count=size)[1:].reshape((-1, 74))
    return decoded

def _ndarrInt_to_CintArray(ndarr: np.ndarray):
    data = ndarr.astype(np.int32)
    data_p = data.ctypes.data_as(c_int_p)
    return data_p, c_int(len(data))