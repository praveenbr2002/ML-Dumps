FROM ubuntu:20.04


RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m bhajji

RUN chown -R bhajji:bhajji /home/bhajji

COPY --chown=bhajji . /home/bhajji/Wrinkle_Detection

USER bhajji

RUN cd /home/bhajji/Wrinkle_Detection && pip3 install -r requirements.txt

WORKDIR /home/bhajji/Wrinkle_Detection

EXPOSE 8080

ENTRYPOINT python3 mainapp.py
