#! /bin/bash

#docker build -t tutorial .
docker run --volume ~/kongmingqi/:/root/kongmingqi -p 0.0.0.0:7000:8888 -it tutorial /bin/bash
