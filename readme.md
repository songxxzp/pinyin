## Pinyin IME Project

项目仓库（为便于检查，截止日期后会设置为public）：https://github.com/songxxzp/pinyin
> git clone git@github.com:songxxzp/pinyin.git

### 项目结构
| dir | statement |
| --- | --------- |
| data | 输入输出文件 |
| src | 源代码 |
| results | 实验结果 |
| models | 概率模型 |

### 运行方式
#### 基本使用方式
```
python src/main.py
```
将按照默认参数运行
#### 传入参数
```
python src/main.py --model-path "models/bigram" --input-file-path interactive

python src/main.py --model-path "models/bigram" --input-file-path "data/input.txt" --output-file-path "data/output.txt"
```
会加载`bigram`模型（使用`sina news`训练的模型），并以交互模式运行。
#### 参数说明
```
python src/main.py -h
```
| 参数 | 说明 |
| ---- | ---- |
| --model-path | 模型路径（文件夹） |
| --output-file-path | 输出文件 |
| --input-file-path | 拼音输入文件，interactive为交互模式，默认为作业测试输入 |
| --std-file-path | 标准答案文件，默认为作业测试标准输出 |
| --vocab-file-path | 词表 |
| --pinyin-file-path | 拼音表 |
| --probability-function | 概率函数，默认为插值 |
| --interpolation-lambda | 差值系数，建议0.03 |
| --top-k-storage | top k |
| --top-k-calculate | top k |
| --max-conditional-prefix-length | 前缀长度 |
| --final-top-k-selector | 最终top k选择器，(std, default, gpt)。std可以计算top k召回率。 |
| --batch-size | for gpt |
| --device | for gpt |
| --lm-tokenizer-path | for gpt |
| --lm-model-path | for gpt |

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
