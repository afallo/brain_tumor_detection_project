.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y brain || :
	@pip install -e .

run_preprocess:
	python -c 'from brain.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from brain.interface.main import train; train()'

#run_pred:
#	python -c 'from brain.interface.main import pred; pred()'

run_evaluate:
	python -c 'from brain.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate
run_all:
	python -c 'from brain.interface.main import main; main()'


run_api:
	uvicorn brain.api.fast:app --reload
