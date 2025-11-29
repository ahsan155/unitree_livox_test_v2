FROM nvcr.io/nvidia/pytorch:25.08-py3
#FROM nvcr.io/nvidia/pytorch:25.01-py3
#FROM nvcr.io/nvidia/pytorch:24.01-py3
#FROM nvcr.io/nvidia/pytorch:24.10-py3

COPY requirements_1.txt requirements_2.txt requirements_3.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements_1.txt


# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

RUN apt-get update && apt-get install -y ros-jazzy-ros-base python3-colcon-common-extensions


# Source setup.bash on container login
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc


RUN ls /usr/local | grep cuda
RUN bash -c "source /opt/ros/jazzy/setup.sh && echo \$ROS_DISTRO"
RUN python3 --version

WORKDIR /
RUN apt-get update && apt-get install -y libffi-dev
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
RUN tar xzf Python-3.10.0.tgz
WORKDIR /Python-3.10.0
RUN pwd
RUN ./configure --enable-optimizations
RUN make -j$(nproc)
RUN make altinstall

RUN python3.10 -m venv /opt/pred_venv
WORKDIR /workspace
RUN /bin/bash -c "source /opt/pred_venv/bin/activate && pip install -r requirements_2.txt"
RUN /bin/bash -c "source /opt/pred_venv/bin/activate && pip install -r requirements_3.txt"

WORKDIR /
COPY jazzy_livox_content /jazzy_livox_content
WORKDIR /usr/src/app
COPY .. ./
RUN /bin/bash -c "source jazzy_livox_content/livox/ws_build/install/setup.bash && python3 setup.py develop"

RUN sudo update
RUN apt install ros-jazzy-rviz2
RUN apt-get update
RUN apt-get install -y libogre-1.12-dev libogre-1.12-dev



