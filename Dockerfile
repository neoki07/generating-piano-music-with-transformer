FROM python:3.7.13-slim

WORKDIR /generating-piano-music-with-transformer
COPY . /generating-piano-music-with-transformer

RUN apt-get update -y
RUN apt-get install curl fluidsynth build-essential libsndfile1 libasound2-dev libjack-dev -y
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update -y
RUN apt-get install google-cloud-sdk -y

RUN mkdir /generating-piano-music-with-transformer/content
RUN mkdir /generating-piano-music-with-transformer/checkpoints

RUN gsutil -q -m cp -r gs://magentadata/models/music_transformer/primers/* /generating-piano-music-with-transformer/content/
RUN gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /generating-piano-music-with-transformer/content/
RUN gsutil -q -m cp -r gs://magentadata/models/music_transformer/checkpoints/* /generating-piano-music-with-transformer/checkpoints

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry export --without-hashes --output requirements.txt
RUN pip install -r requirements.txt
