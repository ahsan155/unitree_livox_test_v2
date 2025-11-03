FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, python3, pip, curl, and other tools
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv python3-dev curl gnupg2 lsb-release build-essential libgl1-mesa-glx

# Setup ROS 2 repository keys and sources
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y ros-humble-ros-base

# Source ROS and setup environment for all users
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Add Nvidia CUDA repository key
# Install wget if not already there
RUN apt-get update && apt-get install -y wget

# Download and install the cuda-keyring
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN rm cuda-keyring_1.1-1_all.deb

# Update apt and install the toolkit (replace with your CUDA version as needed)
RUN apt-get update && apt-get install -y cuda-toolkit-12-4


# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Upgrade pip and install Python dependencies
COPY requirements_1.txt requirements_2.txt requirements_3.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements_1.txt
RUN pip install -r requirements_2.txt
RUN pip install -r requirements_3.txt


# Copy codebase
WORKDIR /usr/src/app
COPY . .

# Build your package if needed
RUN python3 setup.py develop

# Source ROS environment before running
#CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 tools/test_ros_update.py --pt ../pt/livox_model_1.pt"]


# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    build-essential \
    git \
    nano \
    libpcl-dev \
    cmake

# Assuming you're using Ubuntu 22.04 with ROS2 Humble
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-ament-cmake-auto \
    ros-humble-ament-cmake \
    ros-humble-ament-lint-auto \
    ros-humble-ament-cmake-gtest \
    ros-humble-ament-cmake-gmock \
    ros-humble-pcl-conversions \
    ros-humble-pcl-msgs \
    && rm -rf /var/lib/apt/lists/*



# Clone livox driver
#RUN git clone https://github.com/Livox-SDK/livox_ros_driver2.git /workspace/livox_ros_driver2
#WORKDIR /workspace
#RUN ls -la /workspace/livox_ros_driver2
#RUN . /opt/ros/humble/setup.sh && ./livox_ros_driver2/build.sh humble

# Source setup.bash on container login
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

WORKDIR /usr/src/app/eth/test
RUN echo "780	1.0	8.46	3.59" >> test.txt

WORKDIR /usr/livox
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git
WORKDIR /usr/livox/Livox-SDK2
RUN mkdir build
WORKDIR /usr/livox/Livox-SDK2/build
RUN cmake .. && make -j && make install

WORKDIR /usr/livox
RUN git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2
WORKDIR /usr/livox/ws_livox/src/livox_ros_driver2
#RUN ./build.sh humble
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && ./build.sh humble"

CMD ["bash"]
