
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /u/01/shenj4/unix/RA_project/live-speech-reference-search


RUN apt-get update && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/miniconda/bin:$PATH"


# 复制项目文件到工作目录
# COPY . /u/01/shenj4/unix/

# 创建并激活 Conda 虚拟环境
RUN conda env create -f environment0.yml

# 设置环境变量以激活虚拟环境
ENV PATH /opt/conda/envs/live_speech/bin:$PATH


RUN conda install -c conda-forge spacy && \
    python -m spacy download en_core_web_sm


# 暴露端口
EXPOSE 7860

# 启动服务
CMD ["./start.sh", "--daemon"]
# CMD ["gradio", "app.py"]
