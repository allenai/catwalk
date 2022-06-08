.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch catwalk/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ catwalk/
