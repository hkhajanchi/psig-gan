train: 
	python3.5 train.py 

clean: 
	rm -rf  ~/psig-gan/runs/*

gpu_top:
	watch -n0.1 nvidia-smi