train:
	python src/model/train_model.py
dependency:
	python -m pip install -r setup.txt
test:
	python src/test.py
create_data:
	python src/data/make_dataset.py