
## Docker opencv 3 setup

This docker is originally based on the one created by Fabio Stuzt (<a href="https://hub.docker.com/r/flaviostutz/opencv-x86/">here</a>), although it might have suffered customizations throughout time. 
Nonetheless, many thanks to Fabio Stuzt.

Note: this is a big ass docker...

Build the docker:
```
# check if any is running
docker ps
docker build -t opencvtensorflow .
# confirm it is successful
docker images
```
Run Docker and mount code and data directories:

```
# <sudo> docker run -v <volume>:/app -v /tmp:/tmp <image_name> bin/venv <command>
docker run --privileged -p 2222:22 -p 8888:8888 -p 6006:6006 -v ~/Documents/Development/phd/data:/app -v ~/Documents/Development/phd/code:/code  opencvtensorflow
```
This will freeze your terminal. Open another tab and ssh to the docker:
Note: check w ifconfig the ip of the docker.
```
# ssh -X -p 2222 root@[Container IP]
ssh -X -p 2222 root@172.17.0.1
chmod +x /notebooks/run_jupyter.sh 
./notebooks/run_jupyter.sh
```

Note: password is root as well.

To shut down:
```
docker ps
docker kill <container-id>
```