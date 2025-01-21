# conda create -n sovits python==3.10
# conda activate sovits
# git clone https://github.com/RVC-Boss/GPT-SoVITS.git
# cd GPT-SoVITS
# pip install -r requirements.txt
# sudo apt install -y ffmpeg
# rm -rf GPT_SoVITS/pretrained_models/.gitignore
# git lfs install
# git clone https://huggingface.co/lj1995/GPT-SoVITS GPT_SoVITS/pretrained_models
# wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip -O GPT_SoVITS/text/G2PWModel_1.1.zip
# unzip GPT_SoVITS/text/G2PWModel_1.1.zip -d GPT_SoVITS/text/

is_share=True python webui.py zh_CN
