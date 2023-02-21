.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch catwalk/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ catwalk/


.PHONY : docker-testing
docker-testing :
	docker build -t catwalk-testing -f Dockerfile.test .
	beaker image create --workspace ai2/catwalk-tests --name catwalk-testing-tmp catwalk-testing
	beaker image delete petew/catwalk-testing || true
	beaker image rename petew/catwalk-testing-tmp catwalk-testing
