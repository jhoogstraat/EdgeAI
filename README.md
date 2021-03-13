# Build Images
## Tensorflow 2
```
sudo docker build -f Dockerfile.tf2 -t jhoogstraat/edgeai-tf2:latest .
```
## TFLite
```
sudo docker build -f Dockerfile.tflite -t jhoogstraat/edgeai-tflite:latest .
```

# Run Container
## Run with EdgeTPU attached (Integrated and USB)
> Use /dev/video1 when connecting a USB Camera to the Dev Board
```
sudo docker run -it --rm --device /dev/video:/dev/video0 --device /dev/bus/usb:/dev/bus/usb -p 5000:5000 jhoogstraat/edgeai-tflite
```

## Run on Jetson Nano
```
sudo docker run -it --rm --runtime nvidia --device=/dev/video0:/dev/video0 -p 5000:5000 jhoogstraat/edgeai-tf2
```

# Run Server
Execute in container shell
```
MODEL=models/google/tradeoff && \
DETECTOR=pycoral && \
INPUT=/dev/video0 && \
python3 app.py -m $MODEL -d $DETECTOR -v $INPUT
```

# Troubleshooting
- OSError: protocol not found: https://stackoverflow.com/a/40185488/5376091

