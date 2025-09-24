
# Running dots.ocr locally in Mac OS

## 1.Environment
* Hardware: Mac Mini M2, DRAM 24GB
* OS: Sequoia 15.6.1
* [Dots.OCR](https://github.com/rednote-hilab/dots.ocr/)
  * commit id: 75bd78eaca801585453c13e2672edc692c779d6c

## 2.How to Run
* mkdir -p rednote 
* cd rednote && uv init --python=3.12.8
* source ./venv/bin/active
* uv pip install -r requirements.txt
* # link Dots.OCR sub dirs to current one
* ln -s ${Dots.OCR\_DIR}/dots.ocr/{assets,dots\_ocr,weights,tools} .
* python3 tools/download\_model.py # download the model weights
* # patch the ./weights/DotsOCR/configuration\_dots.py
* patch -p0  ./weights/DotsOCR/configuration\_dots.py < use\_qwen2\_vl.diff
* time bash ./run.sh

## 3. NOTE
* the instruction following of Dots.OCR looks not so stable - sometimes the output will not be complete json even asking for it in the chat template, so here forcedly to output the raw text.
* for unknown reason the configuration\_dots.py of the model weights wrongly import Qwen 2.5 VL but what its need is Qwen 2 VL.

## 4. References 
* https://gist.github.com/pllopis/5faf2ecc66ae5d87e3460bf7950511a4
