FROM python:3.6.4-alpine3.7
LABEL maintainer="Felix Last <mail@felixlast.de>"

RUN apk add --no-cache \
    gcc \
    g++ \
    lapack-dev \
    libstdc++ \
    gfortran \
    musl-dev \
    git

RUN pip install numpy && pip install scipy

RUN git clone https://github.com/felix-last/evaluate-kmeans-smote.git /project
WORKDIR /project
RUN pip install -r requirements.txt

RUN mkdir /datasets
RUN mkdir /results

# CMD /bin/bash run_experiment.sh
CMD python remote.py _experiment
