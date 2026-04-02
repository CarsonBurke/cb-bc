# model_constants.py — Single source of truth for model architecture constants.
# Shared between tensor_inference (deploy) and tensor/torch (training).
# No third-party imports — stdlib only.

GRID_SIZE = 5
GRID_HALF = GRID_SIZE // 2
TILE_FEATURES = 21
GLOBAL_FEATURES = 20
H_DIM = 24
HIDDEN_SIZES = [64, 64]
COMPRESS_DIM = 80
RMSNORM_LAYERS = [True, True]
TILE_EMBED_DIM = 8
TILE_INPUT_DIM = TILE_FEATURES + 2  # raw features + (dx/2, dy/2)
TILE_GRID_N = GRID_SIZE * GRID_SIZE

INPUT_SIZE = TILE_GRID_N * TILE_FEATURES + GLOBAL_FEATURES
OBS_SIZE = INPUT_SIZE
TRUNK_INPUT_SIZE = COMPRESS_DIM + GLOBAL_FEATURES + H_DIM
OBS_PREFIX_SIZE = TRUNK_INPUT_SIZE - H_DIM
