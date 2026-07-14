conda env create -f old_env.yml

conda remove pytorch torchvision torchaudio pytorch-mutex

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    pytorch-cuda=11.7 \
    -c pytorch -c nvidia