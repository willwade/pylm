# Adaptive Language Models in Python

> :warning: **A python re-interepretation of the PPM JS Library.** Original found at https://github.com/google-research/google-research/tree/master/jslm - see the original for more code comments. 


This directory contains collection of simple adaptive language models that are
cheap enough memory- and processor-wise to train in a browser on the fly.

## Language Models

### Prediction by Partial Matching (PPM)

Prediction by Partial Matching (PPM) character [language model](ppm_language_model.py). See the bibliography below. If you are looking for alternative implementations 

* **Javascript**:  https://github.com/google-research/google-research/tree/master/jslm
* **C++** https://github.com/pmcharrison/ppm, https://github.com/money6g/ppm
* **Swift** - https://github.com/kdv123/PPMLM
* **Python** (NB: For compression rather than prediction) https://pyppmd.readthedocs.io/en/latest/index.html https://pypi.org/project/pyppmd/


#### Bibliography

1.  Cleary, John G. and Witten, Ian H. (1984): [“Data Compression Using Adaptive Coding and Partial String Matching”](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.4305), IEEE Transactions on Communications, vol. 32, no. 4, pp. 396&#x2013;402.
2.  Moffat, Alistair (1990): [“Implementing the PPM data compression scheme”](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.8728&rep=rep1&type=pdf), IEEE Transactions on Communications, vol. 38, no. 11, pp. 1917&#x2013;1921.
3.  Ney, Reinhard and Kneser, Hermann (1995): [“Improved backing-off for M-gram language modeling”](http://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf), Proc. of Acoustics, Speech, and Signal Processing (ICASSP), May, pp. 181&#x2013;184. IEEE.
4.  Chen, Stanley F. and Goodman, Joshua (1999): [“An empirical study of smoothing techniques for language modeling”](http://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf), Computer Speech &#xff06; Language, vol. 13, no. 4, pp. 359&#x2013;394, Elsevier.
5.  Ward, David J. and Blackwell, Alan F. and MacKay, David J. C. (2000): [“Dasher &#x2013; A Data Entry Interface Using Continuous Gestures and Language Models”](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.3318&rep=rep1&type=pdf), UIST '00 Proceedings of the 13th annual ACM symposium on User interface software and technology, pp. 129&#x2013;137, November, San Diego, USA.
6.  Drinic, Milenko and Kirovski, Darko and Potkonjak, Miodrag (2003): [“PPM Model Cleaning”](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.4389&rep=rep1&type=pdf), Proc. of Data Compression Conference (DCC'2003), pp. 163&#x2013;172. March, Snowbird, UT, USA. IEEE
7.  Jin Hu Huang and David Powers (2004): [“Adaptive Compression-based Approach for Chinese Pinyin Input”](https://www.aclweb.org/anthology/W04-1104.pdf), Proceedings of the Third SIGHAN Workshop on Chinese Language Processing, pp. 24&#x2013;27, Barcelona, Spain. ACL.
8.  Cowans, Phil (2005): [“Language Modelling In Dasher &#x2013; A Tutorial”](http://www.inference.org.uk/pjc51/talks/05-dasher-lm.pdf), June, Inference Lab, Cambridge University (presentation).
9.  Steinruecken, Christian and Ghahramani, Zoubin and MacKay, David (2016): [“Improving PPM with dynamic parameter updates”](https://www.repository.cam.ac.uk/bitstream/handle/1810/254106/Steinruecken%202015%20Data%20Compression%20Conference%202015.pdf), Proc. of Data Compression Conference (DCC'2015), pp. 193&#x2013;202, April, Snowbird, UT, USA. IEEE.
10. Steinruecken, Christian (2015): [“Lossless Data Compression”](https://pdfs.semanticscholar.org/f506/884bb2aefd01ccf3d24a5964aad9ef698679.pdf), PhD dissertation, University of Cambridge.

### Histogram Language Model

Very simple context-less histogram character [language model](histogram_language_model.py).

#### Bibliography

1.  Steinruecken, Christian (2015): [“Lossless Data Compression”](https://pdfs.semanticscholar.org/f506/884bb2aefd01ccf3d24a5964aad9ef698679.pdf), PhD dissertation, University of Cambridge.
2.  Pitman, Jim and Yor, Marc (1997): [“The two-parameter Poisson–Dirichlet distribution derived from a stable subordinator.”](https://projecteuclid.org/download/pdf_1/euclid.aop/1024404422), The Annals of Probability, vol. 25, no. 2, pp. 855&#x2013;900.
3.  Stanley F. Chen and Joshua Goodman (1999): [“An empirical study of smoothing techniques for language modeling”](http://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf), Computer Speech and Language, vol. 13, pp. 359&#x2013;394.

### Pólya Tree (PT) Language Model

Context-less predictive distribution based on balanced binary search trees. Tentative implementation is [here](polya_tree_language_model.py).

#### Bibliography

1.  Gleave, Adam and Steinruecken, Christian (2017): [“Making compression algorithms for Unicode text”](https://arxiv.org/pdf/1701.04047), arXiv preprint arXiv:1701.04047.
2.  Steinruecken, Christian (2015): [“Lossless Data Compression”](https://pdfs.semanticscholar.org/f506/884bb2aefd01ccf3d24a5964aad9ef698679.pdf), PhD dissertation, University of Cambridge.
3.  Mauldin, R. Daniel and Sudderth, William D. and Williams, S. C. (1992): [“Polya Trees and Random Distributions”](https://projecteuclid.org/download/pdf_1/euclid.aos/1176348766), The Annals of Statistics, vol. 20, no. 3, pp. 1203&#x2013;1221.
4.  Lavine, Michael (1992): [“Some aspects of Polya tree distributions for statistical modelling”](https://projecteuclid.org/download/pdf_1/euclid.aos/1176348767), The Annals of Statistics, vol. 20, no. 3, pp. 1222&#x2013;1235.
5.  Neath, Andrew A. (2003): [“Polya Tree Distributions for Statistical Modeling of Censored Data”](http://downloads.hindawi.com/journals/ads/2003/745230.pdf), Journal of Applied Mathematics and Decision Sciences, vol. 7, no. 3, pp. 175&#x2013;186.

## Example

Please see a simple example usage of the model API in [example.py](example.py).

The example has no command-line arguments. To run it using
Python invoke

```shell
> python example.py
```

## Standalone Python Predictor

If you want to try predicting next characters or next words see ``standalone-PPM-predictor\predictor.py``
It trains on the dasher_training.txt and then suggests next letters or words. It caches the word and character models so bewarned if you want to retrain. Just delete the pickle files. 

```shell
python predictor.py  
Processing token: 'hello' with ID: 2
Processing token: 'world' with ID: 3
Final context: 2 -> 3
Node(symbol=None, count=1)
    Node(symbol=2, count=2)
        Node(symbol=3, count=2)
Training character ppm model...
Top 5 character predictions: ['\n', 'q', 'g', 'r', '?']
Training word ppm model...
Top 5 word predictions: ['Tyler.', 'Nicholas.', 'Matthew.', 'Ethan.', 'Aidan.']
```

## Test Utility

A simple test driver [language_model_driver.py](language_model_driver.py) can be used to check that the model behaves using Python 3+. The
driver takes three parameters: the maximum order for the language model, the training file and the test file in text format. Currently only the PPM model is
supported. Note we show in this how you can do next letter and next word predictions **use max_length of around 30** 
Be warned too: training is fast. running test_model can take a long time for the word models. Look at the code - you will need a larger max_length for words_

Example:

```shell
> python language_model_driver.py 30 training_small.txt training_small_test.txt
Results: numSymbols = 54, ppl = 13.268624243648365, entropy = 3.7299468876181376 bits/char
Top 5 character predictions for 'he': ['l', ' ', 'e', 't', 'o']
Results: numSymbols = 54, ppl = 9.575973715690846, entropy = 3.2594191923509106 bits/char
Top 5 word predictions for 'Hello ': ['<OOV>', 'everyone', 'sequence', 'test', 'world']
```

## Example train and test files to use

train.txt

```txt
hello world hello everyone hello there hello world
this is a test this is a trial this is a sequence
welcome to the model test welcome to the world
Gorgeous Doris Day is lovely. One day i went to the beach. 
Today I was at the shops. What day is it today?
```
test.txt

```txt
hello world this is a test sequence
welcome to the test
```


