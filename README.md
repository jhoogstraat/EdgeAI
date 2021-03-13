# Build Images
## TFLite (Raspberry Pi)
```
sudo docker build -f Dockerfile.tflite-arm7l .
```
## TFLite (Coral Dev Board)
```
sudo docker build -f Dockerfile.tflite-aarch64 .
```

# Run Container
## Run with EdgeTPU
```
sudo docker run -it --rm --device /dev/video0:/dev/video0 -v /dev/bus/usb:/dev/bus/usb --privileged -p 5000:5000 $CONTAINER_ID
```

## Run on Jetson Nano
```
sudo docker run -it --rm --runtime nvidia --network host --device=/dev/video0:/dev/video0 $CONTAINER_ID
```