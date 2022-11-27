FROM tensorflow/tensorflow:latest-gpu

# nvidia key stuff
RUN apt-key del 7fa2af80

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# necessary for tensorflow
RUN apt-get update
RUN apt-get install -y xvfb ffmpeg freeglut3-dev

WORKDIR /tf_agent

COPY requirements.txt requirements.txt
RUN python --version
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN pwd && ls -la

CMD ["python", "main.py"]