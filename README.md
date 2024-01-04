# NLP
This repository contains code and data for various natural language processing (NLP) tasks, such as text summarization, sentiment analysis, and machine translation.

## Requirements
- Python 3.8 or higher
- PyTorch 1.8 or higher
- Transformers 4.5 or higher
- NLTK 3.6 or higher
- Spacy 3.0 or higher

## Installation
- Clone this repository: `git clone https://github.com/shfarhaan/NLP.git`
- Install the dependencies: `pip install -r requirements.txt`
- Download the pretrained models: `python download_models.py`

## Usage
- To run text summarization, use: `python summarize.py --input input.txt --output output.txt --model t5-base`
- To run sentiment analysis, use: `python sentiment.py --input input.txt --output output.txt --model bert-base-cased`
- To run machine translation, use: `python translate.py --input input.txt --output output.txt --model m2m-100 --src_lang en --tgt_lang fr`

## License
This project is licensed under the MIT License - see the LICENSE file for details.
