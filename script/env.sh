conda activate base
conda remove --name ecg --all --yes

# 创建环境
conda create --name ecg python=3.9.20 --yes
conda activate ecg

# 安装PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# 安装加载和处理预训练模型的库
conda install conda-forge::transformers=4.46.3 --yes

# 安装数组形状的库
conda install conda-forge::einops=0.8.0 --yes

# 安装easydict库
conda install conda-forge::easydict=1.13 --yes

# 安装计算和显示Grad-CAM热图的库
pip install grad-cam==1.5.4
