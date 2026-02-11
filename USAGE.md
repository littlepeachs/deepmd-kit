pip install -e .
cd test_mptraj
dp --pt train --skip-neighbor-stat input_dpa3.json
当前is_debug模式模拟单卡跑四个rank的图拆分