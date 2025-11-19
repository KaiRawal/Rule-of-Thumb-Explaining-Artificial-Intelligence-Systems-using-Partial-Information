
cd CODE

git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
cd OpenXAI
git checkout 2bae071737bddf0bfac2b2714964f08b996a8ab1
git apply --whitespace=nowarn ../patch.diff

cd ..
mv OpenXAI/openxai ./openxai_inner
rm -rf OpenXAI
mv ./openxai_inner ./openxai

cd ..

mkdir ./PLOTS
mkdir ./DATA
mkdir ./DATA/PLOTS
mkdir ./DATA/EVALUATOR_OUTPUTS
mkdir ./DATA/EVALUATOR_OUTPUTS/META
mkdir ./DATA/EVALUATOR_OUTPUTS/PLOTS



python3 CODE/gen_explanations.py --model_name lr --data_name compas
# python3 -W ignore::FutureWarning ./CODE/gen_auc_plots.py --data_name compas

python3 CODE/gen_explanations.py --model_name lr --data_name german
# python3 -W ignore::FutureWarning ./CODE/gen_auc_plots.py --data_name german

python3 CODE/gen_explanations.py --model_name lr --data_name heloc
# python3 -W ignore::FutureWarning ./CODE/gen_auc_plots.py --data_name heloc

python3 CODE/gen_explanations.py --model_name lr --data_name adult
# python3 -W ignore::FutureWarning ./CODE/gen_auc_plots.py --data_name adult


python3 CODE/gen_explanations.py --model_name ann --data_name german
python3 CODE/gen_explanations.py --model_name ann --data_name compas
python3 CODE/gen_explanations.py --model_name ann --data_name heloc
python3 CODE/gen_explanations.py --model_name ann --data_name adult
