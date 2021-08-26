echo Starting experiments
pause

call conda activate ml
python main.py -p "experiments/mnist_ali_robust.json" -o "experiments"

::python main.py -p "experiments/mnist_ali_full.json" -o "experiments"
::python main.py -p "experiments/mnist_ali_lvl0.json" -o "experiments"
::python main.py -p "experiments/mnist_ali_lvl1.json" -o "experiments"
::python main.py -p "experiments/mnist_ali_lvl2.json" -o "experiments"
::python main.py -p "experiments/mnist_ali_lvl3.json" -o "experiments"

::python main.py -p "experiments/svhn_ali_full.json" -o "experiments"
::python main.py -p "experiments/svhn_ali_lvl0.json" -o "experiments"
::python main.py -p "experiments/svhn_ali_lvl1.json" -o "experiments"
::python main.py -p "experiments/svhn_ali_lvl2.json" -o "experiments"
::python main.py -p "experiments/svhn_ali_lvl3.json" -o "experiments"

::python main.py -p "experiments/cifar_ali_full.json" -o "experiments"
::python main.py -p "experiments/cifar_ali_lvl0.json" -o "experiments"
::python main.py -p "experiments/cifar_ali_lvl1.json" -o "experiments"
::python main.py -p "experiments/cifar_ali_lvl2.json" -o "experiments"
::python main.py -p "experiments/cifar_ali_lvl3.json" -o "experiments"

python main.py -p "experiments_imagenet/experiment1.json" -o "experiments_imagenet" -z "experiments_imagenet/encoded_z_0.npy"

echo Experiments complete
pause