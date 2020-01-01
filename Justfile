
prepare:
	#!/bin/bash
	cd prepare_training_data
	cargo run --release > ../traindat.json

create-model:
    #!/bin/bash
	pipenv run ./create_model.py

train:
	#!/bin/bash
	echo TODO
	exit 1
