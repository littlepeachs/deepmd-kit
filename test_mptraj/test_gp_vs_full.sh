#!/bin/bash

# 测试 GP 模式 vs 全图模式
# 用法: bash test_gp_vs_full.sh

INPUT_JSON="input_dpa3.json"

echo "========================================"
echo "测试 GP 模式 vs 全图模式"
echo "========================================"

# 1. 运行 GP 模式
echo ""
echo "[1/2] 运行 GP 模式..."
echo "========================================"


# 设置环境变量：启用 GP 模式
export DISABLE_GP_MODE=0

dp --pt train --skip-neighbor-stat input_dpa3.json


# 2. 运行全图模式
echo ""
echo "[2/2] 运行全图模式..."
echo "========================================"


# 设置环境变量：禁用 GP 模式
export DISABLE_GP_MODE=1

# 运行训练
echo "开始训练（全图模式）..."
dp --pt train --skip-neighbor-stat $INPUT_JSON
