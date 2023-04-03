## Pinyin IME Project

项目仓库（为便于检查，截止日期后会设置为public）：https://github.com/songxxzp/pinyin
> git clone git@github.com:songxxzp/pinyin.git

### 项目结构
| 目录 | 说明 |
| --- | --------- |
| data | 输入输出文件 |
| src | 源代码 |
| results | 实验结果 |
| models | 概率模型 |

### 运行方式
#### 基本使用方式
```bash
python src/main.py
```
将按照默认参数运行

#### 传入参数
```bash
python src/main.py --model-path "models/bigram" --input-file-path interactive
# 会加载`bigram`模型（使用`sina news`训练的模型），并以交互模式运行。

python src/main.py --model-path "models/bigram" --input-file-path "data/input.txt" --output-file-path "data/output.txt"
# 会加载`bigram`模型，根据指定输入输出运行。
```

#### 运行实验
```bash
python src/experiment.py
# 运行src/experiment.py中的预设实验，结果在results中
```

#### 参数说明
```bash
python src/main.py -h
```
| 参数 | 说明 |
| ---- | ---- |
| --model-path | 模型路径（文件夹） |
| --output-file-path | 输出文件 |
| --input-file-path | 拼音输入文件，interactive为交互模式，默认为作业测试输入 |
| --std-file-path | 标准答案文件，默认为作业测试标准输出 |
| --vocab-file-path | 词表，默认即可 |
| --pinyin-file-path | 拼音表，默认即可 |
| --probability-function | 概率函数平滑方式，默认为插值(interpolation)，可选拉普拉斯平滑(laplace)，建议默认。 |
| --interpolation-lambda | 差值系数，建议0.03，默认即可。 |
| --top-k-storage | top k，默认为3，建议两者一致。 |
| --top-k-calculate | top k，默认为3，建议两者一致。 |
| --max-conditional-prefix-length | 前缀长度，默认为2，会根据模型调整，默认即可。 |
| --final-top-k-selector | 最终top k选择器，(std, default, gpt)。std可以计算top k召回率，default选择概率最高者，gpt选择ppl最低者。 |
| --batch-size | for gpt |
| --device | for gpt, cpu or cuda(default) |
| --lm-tokenizer-path | for gpt |
| --lm-model-path | for gpt, default 'uer/gpt2-chinese-cluecorpussmall' |

文件目录建议填写绝对路径，或者相对main.py的路径。

### 构建模型
1. 新建文件夹: `models/trigram_custom`

2. 构建`config.json`
    ```json
    {
        "max_prefix_length": 2,
        "corpora_path": [
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-02.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-04.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-05.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-06.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-07.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-08.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-09.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-10.txt", "gbk", ["html", "title"]],
            ["/home/song/下载/拼音输入法作业-2023春/语料库/语料库/sina_news_gbk/2016-11.txt", "gbk", ["html", "title"]],
            ["/home/song/data/news_crawl", "utf-8", []]
        ]
    }
    ```

    其中`corpora_path`中元素由三元组组成`(path, format, labels)`。若`labels`为`[]`，则将语料作为`txt`处理，否则作为`jsonl`处理，并处理`labels`中的内容。`path`若为一目录，将处理其下全部文件。

3. 运行`python src/main.py --model-path "models/trigram_custom"`。若不存在以构建的模型，会根据`models/trigram_custom`目录下的`config.json`自行构建模型。在构建`trigram`模型时，请保证内存充足。

4. `models`中包括了部分预设的`config.json`，请更改语料路径后使用。也可下载已构建好的模型：https://cloud.tsinghua.edu.cn/d/cffa2e2502ed4fd59d1b/

### 额外使用数据集：
nlp_chinese_corpus : https://github.com/brightmart/nlp_chinese_corpus
news-crawl (From WMT): https://data.statmt.org/news-crawl/zh/

### 模型下载：
https://github.com/songxxzp/Pinyin-IME-Models
