
# catalyst-dynamic-text-classification

This code is a re-implementation of the paper [Metric Learning for Dynamic Text Classification](https://arxiv.org/abs/1911.01026) by ASAPP Research using [Catalyst](https://github.com/catalyst-team/catalyst) framework.

The original code for the paper can be found here [asappresearch/dynamic-classification](https://github.com/asappresearch/dynamic-classification).


## How to run

1.  Clone repository

		git clone git@github.com:xelibrion/catalyst-dynamic-text-classification.git
        cd catalyst-dynamic-text-classification

2.  Install dependencies

    	pip install -e .

3.  Fetch data

		cd dynamic_class
		./get_data.py

4.  Run train script to build vocabulary (it will fail to train the model without embeddings)

	    ./train.py

5.  Compute words vectors for the vocabulary using a fasttext model. Can be downloaded [here](https://fasttext.cc/docs/en/crawl-vectors.html).

		cat input/vocab.txt | awk -F ' ' '{print $1}' > vocab_words.txt
		~/projects/fasttext/fasttext print-word-vectors  ~/projects/fasttext/cc.en.300.bin < vocab_words.txt > vocab_vectors.txt

	Please note that the original paper used GloVe as word embeddings. You might want to experiment with the choice of embeddings.

	Also, the tokenizer could be much better - at the moment it simply splits on whitespace.


6.  Train the model

    	./train.py

## Gotchas

This pipeline uses [**sru** package](https://github.com/asappresearch/sru/issues/109#issuecomment-649413371), which might cause some challenges to get things running. See my comment [here](https://github.com/asappresearch/sru/issues/109#issuecomment-649413371).
