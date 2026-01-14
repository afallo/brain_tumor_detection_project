.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

clean:
	 rm -rf build dist *.egg-info

reinstall_package: clean
	 @pip uninstall -y brain || :
	 @pip install -e .


run_preprocess_classification:
	python -c 'from brain.interface.main import preprocess_classification; preprocess_classification()'

run_train_classification:
	python -c 'from brain.interface.main import train_classification; train_classification()'

#run_pred:
#	python -c 'from brain.interface.main import pred; pred()'

run_evaluate_classification:
	python -c 'from brain.interface.main import evaluate_classification; evaluate_classification()'

# run_all: run_preprocess run_train run_pred run_evaluate
run_all_classification:
	python -c 'from brain.interface.main import main_classification; main_classification()'


run_preprocess_seg2D:
	python -c 'from brain.interface.main import preprocess_seg2D; preprocess_seg2D()'

run_train_seg2D:
	python -c 'from brain.interface.main import train_seg2D; train_seg2D()'

#run_pred:
#	python -c 'from brain.interface.main import pred; pred()'

run_evaluate_seg2D:
	python -c 'from brain.interface.main import evaluate_seg2D; evaluate_seg2D()'

# run_all: run_preprocess run_train run_pred run_evaluate
run_all_seg2D:
	python -c 'from brain.interface.main import main_seg2D; main_seg2D()'



run_preprocess_seg3D:
	python -c 'from brain.interface.main import preprocess_seg3D; preprocess_seg3D()'


run_api:
	uvicorn brain.api.fast:app --reload

run_api_local:
	uvicorn brain.api.fast_local:app --reload
