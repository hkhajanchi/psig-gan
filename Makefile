train1: 
	python3.5 train1.py

train2: 
	python3.5 train2.py

train3: 
	python3.5 train3.py

train_d:
	python3.5 train_discrim.py

clean: 
	rm -rf  ~/psig-gan/runs/*

gpu_top:
	watch -n0.1 nvidia-smi