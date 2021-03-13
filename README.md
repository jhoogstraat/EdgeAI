# Build Images
## Tensorflow 2
```
sudo docker build -f Dockerfile.tf2 .
```
## TFLite
```
sudo docker build -f Dockerfile.tflite .
```

# Run Container
## Run with EdgeTPU attached (Integrated and USB)
```
sudo docker run -it --rm --device /dev/video0:/dev/video0 -v /dev/bus/usb:/dev/bus/usb --privileged -p 5000:5000 $CONTAINER_ID
```

## Run on Jetson Nano
```
sudo docker run -it --rm --runtime nvidia --device=/dev/video0:/dev/video0 -p 5000:5000 $CONTAINER_ID
```