echo Starting experiments
pause

call conda activate ml
python main.py -p "experiments/mnist_ali_full.json" -o "results"
python main.py -p "experiments/mnist_ali_lvl0.json" -o "results"
python main.py -p "experiments/mnist_ali_lvl1.json" -o "results"
python main.py -p "experiments/mnist_ali_lvl2.json" -o "results"
python main.py -p "experiments/mnist_ali_lvl3.json" -o "results"

python main.py -p "experiments/svhn_ali_full.json" -o "results"
python main.py -p "experiments/svhn_ali_lvl0.json" -o "results"
python main.py -p "experiments/svhn_ali_lvl1.json" -o "results"
python main.py -p "experiments/svhn_ali_lvl2.json" -o "results"
python main.py -p "experiments/svhn_ali_lvl3.json" -o "results"

python main.py -p "experiments/cifar_ali_full.json" -o "results"
python main.py -p "experiments/cifar_ali_lvl0.json" -o "results"
python main.py -p "experiments/cifar_ali_lvl1.json" -o "results"
python main.py -p "experiments/cifar_ali_lvl2.json" -o "results"
python main.py -p "experiments/cifar_ali_lvl3.json" -o "results"

echo Experiments complete
pause