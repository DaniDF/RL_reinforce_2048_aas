#!/bin/bash

chmod u+x setup.sh
chmod u+x train.sh

if [[ ! -f "/tmp/py2048_setup" || $1 == "setup" ]] ;
then
	echo "Setting up..."

	touch "/tmp/py2048_setup"

	./setup.sh
fi

nohup ./train.sh
