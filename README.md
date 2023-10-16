# Khan-license-plate-recognition

The Docker File in the system has all the libraries and necessary dependencies required for installation.
#
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

#
WORKDIR /code

#
ENTRYPOINT []

# Install gst extenstions and opencv dependencies
RUN apt-get update && \
    apt install python3-opencv -y && \
    apt install python3-numpy -y && \
    apt install python3-gi python3-dev python3-gst-1.0 -y && \
    apt install libgstrtspserver-1.0-0 gstreamer1.0-rtsp -y && \
    apt install libgirepository1.0-dev -y && \
    apt install gobject-introspection gir1.2-gst-rtsp-server-1.0 -y  && \
    apt install libpq-dev


# Install pyds and extra dependencies
RUN /opt/nvidia/deepstream/deepstream-6.3/user_additional_install.sh
RUN /opt/nvidia/deepstream/deepstream-6.3/user_deepstream_python_apps_install.sh -b
