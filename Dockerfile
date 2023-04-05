FROM ubuntu:latest
ARG EARTHDATA_USER
ARG EARTHDATA_PASSWORD

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get install -y gdal-bin libgdal-dev
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3-pip

RUN pip install torch sentinelsat rasterio gdal-utils torchvision

RUN apt-get -y install git
WORKDIR /repo
ADD ./requirements.txt .
RUN pip install -r requirements.txt

#ARG CYAN_APP_KEY
#ARG ESA_PASSWORD1
#ARG ESA_USER1
#ARG ESA_PASSWORD2
#ARG ESA_USER2
#ARG SAVE_FOLDER

RUN echo "machine urs.earthdata.nasa.gov login $EARTHDATA_USER password $EARTHDATA_PASSWORD" > ~/.netrc ; > ~/.urs_cookies
RUN chmod  0600 ~/.netrc
RUN echo $EARTHDATA_USER


#ADD ./run.sh .
#CMD ["./run.sh"]
CMD ["python3", "-u", "/home/conrad/hab_detection/dataset_create/create_dataset.py"]

