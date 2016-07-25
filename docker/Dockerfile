FROM ubuntu:14.04

RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ trusty multiverse" >> /etc/apt/sources.list \
     && apt-get update \
     && apt-get -y install libopencv-dev build-essential cmake git \
           libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 \
           libdc1394-22-dev libjpeg-dev libpng12-dev libtiff4-dev \
           libjasper-dev libavcodec-dev libavformat-dev libswscale-dev \
           libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
           libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev \
           libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev \
           libvorbis-dev libxvidcore-dev x264 v4l-utils unzip ssh openssh-server

RUN cd /opt \
     && wget https://github.com/Itseez/opencv/archive/3.0.0.zip \
     && unzip 3.0.0.zip \
     && cd opencv-3.0.0 \
     && mkdir release \
     && cd release \
     && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON .. \
     && make -j4 \
     && make install \
     && bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' \
     && ldconfig \
     && rm /opt/3.0.0.zip \
     && rm -R /opt/opencv-3.0.0

RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && echo "export VISIBLE=now" >> /etc/profile

# SSH login fix. Otherwise user is kicked off after login
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"

CMD ["/usr/sbin/sshd", "-D"]



#### PART 2 - setup tensorflow

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-numpy \
        python-pip \
        python-scipy \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        && \
    python -m ipykernel.kernelspec

ENV TENSORFLOW_VERSION 0.8.0

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
#COPY run_jupyter.sh /jupyter

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"
#CMD ["chmod +x /notebooks/run_jupyter.sh"]
#CMD ["/notebooks/run_jupyter.sh"]

EXPOSE 22