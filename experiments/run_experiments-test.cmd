echo Starting experiments
pause

call conda activate ml

python main.py -p "experiments/svhn_ali_full.json" -o "experiments"
python main.py -p "experiments/svhn_ali_lvl0.json" -o "experiments"
python main.py -p "experiments/svhn_ali_lvl1.json" -o "experiments"
python main.py -p "experiments/svhn_ali_lvl2.json" -o "experiments"
python main.py -p "experiments/svhn_ali_lvl3.json" -o "experiments"

echo Experiments complete
pause