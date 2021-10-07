SHELL:=/bin/bash

run-docker:
	sudo docker run -it --rm --device /dev/video0:/dev/video0 -v /dev/bus/usb:/dev/bus/usb --privileged -p 5000:5000 dd09047cad8d

run-g-accuracy:
	source .env/bin/activate; \
	python3 app.py -m models/google/accuracy -d pycoral -v /dev/video0

run-g-tradeoff:
	source .env/bin/activate; \
	python3 app.py -m models/google/tradeoff -d pycoral -v /dev/video0

run-ms:
	source .env/bin/activate; \
	python3 app.py -m models/ms -d ms-tflite -v /dev/video0

.PHONY: run-docker, run-g-accuracy, run-g-tradeoff, run-ms