FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install python3 -y
RUN apt-get install pip -y
RUN apt-get install python3-tk -y
RUN pip install numpy
RUN pip install pillow
RUN pip install onnxruntime
RUN pip install onnxruntime-gpu
RUN pip install mediapipe
RUN pip install playsound
RUN pip install tk
RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torchvision
RUN pip install tensorflow==2.7.0
RUN pip install tensorflow-gpu==2.8.0
ARG USER
RUN groupadd -g 1000 $USER
RUN useradd -d /home/$USER -s /bin/bash -m $USER -u 1000 -g 1000
USER $USER
ENV HOME /home/$USER

WORKDIR /home/data
COPY . .
CMD [ "python3", "codes/main_ui.py" ]
