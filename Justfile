
prepare:
	#!/bin/bash
	cd prepare_training_data
	cargo run --release | pigz > ../traindat.json.gz

create:
	#!/bin/bash
	pipenv run ./create_model.py

train:
	#!/bin/bash
	pipenv run ./train_model.py

predict audiofile:
	#!/bin/bash
	abs=$(realpath "{{audiofile}}")
	(cd predict && cargo run --release -- "${abs}") | pipenv run ./predict.py
