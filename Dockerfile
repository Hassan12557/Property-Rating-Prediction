FROM ubuntu:latest
LABEL authors="TDKS"

ENTRYPOINT ["top", "-b"]