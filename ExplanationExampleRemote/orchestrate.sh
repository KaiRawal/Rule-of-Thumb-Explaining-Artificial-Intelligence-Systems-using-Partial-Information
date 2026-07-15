#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/DATA"

curl -L -o "$SCRIPT_DIR/DATA/dog-and-cat-classification-dataset.zip" https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset
unzip -o "$SCRIPT_DIR/DATA/dog-and-cat-classification-dataset.zip" -d "$SCRIPT_DIR/DATA"

cp "$SCRIPT_DIR/DATA/PetImages/Cat/12499.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/23.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12498.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/391.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12497.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/445.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12496.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/910.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12495.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1151.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12494.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1267.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12493.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1386.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12492.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1757.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12491.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1773.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12480.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1914.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12489.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1936.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12488.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/1937.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12487.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/2021.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Cat/12486.jpg" "$SCRIPT_DIR/DATA/PetImages/Cat/2189.jpg" # corrupt image file

cp "$SCRIPT_DIR/DATA/PetImages/Dog/12499.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/50.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12498.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/296.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12497.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/414.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12496.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/522.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12495.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/543.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12494.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/561.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12493.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/565.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12492.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/573.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12491.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/663.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12490.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/719.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12489.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1017.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12488.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1168.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12487.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1259.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12486.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1356.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12485.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1884.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12484.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1900.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12483.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1914.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12482.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1936.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12481.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/1937.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12480.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/2021.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12479.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/2317.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12478.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/2353.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12477.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/2479.jpg" # corrupt image file
cp "$SCRIPT_DIR/DATA/PetImages/Dog/12476.jpg" "$SCRIPT_DIR/DATA/PetImages/Dog/2494.jpg" # corrupt image file



if [ -f "$SCRIPT_DIR/openai_classification_results.csv" ]; then
    cp "$SCRIPT_DIR/openai_classification_results.csv" "$SCRIPT_DIR/DATA/"
    echo "Using existing OpenAI classification results."
else
    python "$SCRIPT_DIR/classify_openai.py" \
        --num-images 5000 \
        --model gpt-4o-mini \
        --results-csv "$SCRIPT_DIR/DATA/openai_classification_results.csv"
fi

python "$SCRIPT_DIR/run.py"

mkdir -p "$SCRIPT_DIR/ExampleImages"
cp "$SCRIPT_DIR/DATA/results/saliency_2_c.pdf" "$SCRIPT_DIR/ExampleImages/"
cp "$SCRIPT_DIR/DATA/results/saliency_224_c.pdf" "$SCRIPT_DIR/ExampleImages/"
cp "$SCRIPT_DIR/DATA/results/saliency_6_d.pdf" "$SCRIPT_DIR/ExampleImages/"
cp "$SCRIPT_DIR/DATA/results/saliency_117_d.pdf" "$SCRIPT_DIR/ExampleImages/"
cp "$SCRIPT_DIR/DATA/results/saliency_1450_c.pdf" "$SCRIPT_DIR/ExampleImages/"
cp "$SCRIPT_DIR/DATA/results/saliency_1773_d.pdf" "$SCRIPT_DIR/ExampleImages/"

