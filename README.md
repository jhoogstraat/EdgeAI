# Build Images
## TFLite
```
sudo docker build -f Dockerfile.tflite -t jhoogstraat/edgeai-tflite:latest .
```
## Tensorflow 2
```
sudo docker build -f Dockerfile.tf2 -t jhoogstraat/edgeai-tf2:latest .
```

# Run Container
## Run with EdgeTPU attached (Integrated and USB)
> Use /dev/video1 when connecting a USB Camera to the Coral Dev Board
```
sudo docker run -it --rm --privileged -p 5000:5000 jhoogstraat/edgeai-tflite -m $MODEL -d $DETECTOR -v $INPUT
```

## Run on Jetson Nano
```
sudo docker run -it --rm --runtime nvidia --privileged -p 5000:5000 jhoogstraat/edgeai-tf2 -m $MODEL -d $DETECTOR -v $INPUT
```

# Troubleshooting
- OSError: protocol not found: https://stackoverflow.com/a/40185488/5376091

