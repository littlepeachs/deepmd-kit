export DISABLE_GP_MODE=1
CUDA_VISIBLE_DEVICES=0 DP_GP_PDB_RANKS=0 dp --pt train --skip-neighbor-stat ./test_mptraj/input_dpa3.json