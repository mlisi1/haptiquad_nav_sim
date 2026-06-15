FROM anymal_sim_base:latest


USER root
ARG USERNAME=anydev


RUN rm -f /etc/apt/auth.conf.d/*.conf || true \
 && rm -f /etc/apt/sources.list.d/*anybotics* || true \
 && rm -f /etc/apt/preferences.d/*anybotics* || true \
 && apt-get update || true

# Set up sudo
RUN apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME 
  # && rm -rf /var/lib/apt/lists/*

USER $USERNAME
# optional: ensure bash is default
SHELL ["/usr/bin/bash", "-c"]

# example: install extra tools (customize as needed)
RUN sudo apt-get install -y \
    curl \
    git \
    nano \ 
    mesa-utils \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libc++-dev \
    libc++abi-dev \
    python3-tk \
    python3-pip \
    python3-pil.imagetk \
    ros-noetic-robot-state-publisher \
    ros-noetic-pinocchio \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-plotjuggler-ros \
    ros-noetic-plotjuggler \
    ros-noetic-move-base \
    ros-noetic-nav-core \
    ros-noetic-costmap-2d \
    ros-noetic-navfn \
    ros-noetic-base-local-planner \
    ros-noetic-dwa-local-planner \
    ros-noetic-rotate-recovery \
    ros-noetic-clear-costmap-recovery \
    ros-noetic-map-server 

# optional workspace directory
WORKDIR /home/anydev/anymal_ws

RUN echo 'source /home/anydev/anymal_ws/devel/setup.bash' >> ~/.bashrc


# keep default behavior similar to your run command
CMD ["/usr/bin/bash"]