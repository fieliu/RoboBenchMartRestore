
# 环境
conda 安装python 3.10
pip 直接安装了requirements文件了

下载两个资产

一个用于商场里面的物品
RoboBenchMart_assets
hf download emb-ai/RoboBenchMart_assets --repo-type dataset --local-dir assets


一个用于商场环境
mkdir -p /home/lh/.maniskill/data/scene_datasets && wget -c --progress=bar:force --timeout=60 --tries=5 "https://hf-mirror.com/datasets/haosulab/RoboCasa/resolve/main/robocasa_dataset.zip" -O /home/lh/.maniskill/data/scene_datasets/robocasa_dataset.zip