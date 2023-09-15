![AIO](imgs/aio.png)

- [AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)

## 更新履歴
- 2023/09/15: 本ベースラインを公開しました。


## 目次

- Open-Domain QA
- 初めに
- ディレクトリ構造
- 学習済みモデルのダウンロード
- 環境構築
  - コンテナの起動
- [Dense Passage Retrieval](#retriever-dense-passage-retrieval)
    - データセット
      - ダウンロード
    - Retriever
      - 設定
      - データセットの質問に関連する文書の抽出
- [Fusion-in-Decoder](#reader-fusion-in-decoder)
    - データセット
      - 作成
      - 形式
    - Reader
      - 解答生成と評価
- 謝辞・ライセンス



# Open-Domain QA

本実装では、早押し形式のオープンドメイン質問応答に取り組むために二つのモジュール（Retriever-Reader）を使用します。<br>
1. 与えられた質問に対して、文書集合から関連する文書を検索するモジュール（Retriever - Dense Passage Retrieval）
2. 質問と検索した関連文書から、質問の答えを生成するモジュール（Reader - Fusion-in-Decoder）

より詳細な解説は、以下を参照して下さい。

> Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau. Dense Passage Retrieval for Open-Domain Question Answering (EMNLP2020) [\[paper\]](https://www.aclweb.org/anthology/2020.emnlp-main.550) [\[github\]](https://github.com/facebookresearch/DPR)

> Gautier, Izacard and Edouard, Grave. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (EACL2021) [\[paper\]](https://aclanthology.org/2021.eacl-main.74.pdf)


## 初めに
本実装は、通常のオープンドメイン質問応答形式による学習を行ったモデルを、出力の生成確率に基づいて早押し形式での解答に対応させたものです。

従って、早押し自体の学習を行ったモデルではありません。

<br>
モデルを動かすにあたって、まず本リポジトリのクローンとディレクトリの移動を行ってください。

```bash
# コマンド実行例
$ git clone git@github.com:cl-tohoku/aio4-fid-baseline.git
$ cd aio4-fid-baseline
```


## ディレクトリ構造

```yaml
- datasets.yml:                        データセット定義

# データセットの前処理
- prepro/:
  - convert_dataset.py:                データ形式変換

# Retriever
- retrievers/:
  - DPR/:                              DPR モジュール

# 生成型 Reader
- generators/:
  - fusion_in_decoder/:                FiD モジュール
  
- download_models.sh:                  学習済みモデルのダウンロードスクリプト
- compute_score.py:                    評価スクリプト
```

## 学習済みモデルのダウンロード
本ベースラインでは、クイズに対する関連文書を検索するためのDPR(Dense Passage Retrieval)と、検索された関連文書から解答を生成するためのFiD(Fusion-in-Decoder)の学習済みモデルの配布を行っています。

学習済みモデルは下記のコマンドでダウンロードすることができますが、その際に gcloud CLI コマンド を使用するため、Google Cloud CLI のインストールが必要です。

そのため、下記ダウンロードコマンドを実行する際には、事前に以下のページの指示に従い、お使いの環境に応じて Google Cloud CLI をインストールしてください。

https://cloud.google.com/sdk/docs/install?hl=ja

```bash
# 学習済みモデルのダウンロード
$ bash download_models.sh
$ du -h retrievers/DPR/models/baseline/*
  2.5G    biencoder.pt
  13G     embedding.pickle
$ du -h generators/fusion_in_decoder/models_and_results/baseline/*
  4.0K    config.json
  1.7G    optimizer.pth.tar
  851M    pytorch_model.bin
```


## 環境構築
### Dockerコンテナの起動

- まず、Dockerコンテナを起動します。
```bash
# コマンド実行例
$ docker image build --tag aio4_fid:latest .
$ docker container run \
      --name fid_baseline \
      --rm \
      --interactive \
      --tty \
      --gpus all \
      --mount type=bind,src=$(pwd),dst=/app \
      aio4_fid:latest \
      bash
```


## Retriever (Dense Passage Retrieval)

![retriever](imgs/retriever.png)


## データセット

- Retriever (Dense Passage Retrieval) の訓練データには、クイズ大会[「abc/EQIDEN」](http://abc-dive.com/questions/) の過去問題に対して Wikipedia の記事段落の付与を自動で行ったものを使用しています。

- 以上のデータセットの詳細については、[AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/) の公式サイト、および下記論文をご覧下さい。

> __JAQKET: クイズを題材にした日本語QAデータセット__
> - https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
> - 鈴木正敏, 鈴木潤, 松田耕史, ⻄田京介, 井之上直也. JAQKET:クイズを題材にした日本語QAデータセットの構築. 言語処理学会第26回年次大会(NLP2020) [\[PDF\]](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)


### ダウンロード
第4回AI王コンペティションで配布されている訓練・開発・リーダーボード評価用クイズ問題、およびRetriever (Dense Passage Retrieval) の学習で使用するデータセット（訓練・開発用クイズ問題に Wikipedia の記事段落の付与を行ったもの）は、下記のコマンドで取得することができます。
<br>

```bash
$ cd retrievers/DPR
$ datasets_dir="datasets"
$ bash scripts/download_data.sh $datasets_dir
```

```bash
# ダウンロードされたデータセット
<datasets_dir>
|- aio_02_train.jsonl                    # 第4回DPR訓練データ（第2回のものと同じものです）
|- aio_04_dev_unlabeled_v1.0.jsonl       # 第4回開発データ（問題のみ）
|- aio_04_dev_v1.0.jsonl                 # 第4回開発データ（問題と正解）
|- aio_04_test_lb_unlabeled_v1.0.jsonl   # 第4回リーダーボード用データ
|- wiki/
|  |- jawiki-20220404-c400-large.tsv.gz  # Wikipedia 文書集合
|- retriever/
|  |- aio_02_train.json.gz               # 第4回訓練データに Wikipedia の記事段落の付与を行ったもの
|  |- aio_02_train.tsv                   # 「質問」と「正解」からなる TSV 形式の訓練データ
```

| データ         | ファイル名                                    |    質問数 |       文書数 |
|:------------|:-----------------------------------------|-------:|----------:|
| 訓練          | aio\_02\_train                           | 22,335 |         - |
| 開発          | aio\_04\_dev\_v1.0.jsonl                 |    500 |         - |
| リーダーボード用データ | aio\_04\_test\_lb\_unlabeled\_v1.0.jsonl |    500 |         - |
| 文書集合        | jawiki-20220404-c400-large               |      - | 4,288,199 |

- データセットの構築方法の詳細については、[retrievers/DPR/data/README.md](retrievers/DPR/data/README.md)を参照して下さい。


## DPRモデルについて
- [学習済みモデルのダウンロード](#学習済みモデルのダウンロード)節で既に学習済みRetriever(DPR)・文書エンベッディングのダウンロードを行った場合は、既にモデルの準備が完了しているため、この節はスキップしてください。
- なお、Retriever(DPR) の学習を行う場合、また文書集合（Wikipedia）のエンコード方法の詳細については、[retrievers/DPR/README.md](retrievers/DPR/README.md)を参照して下さい。

```bash
# 学習済みRetriever(DPR)・文書エンベッディングのダウンロードを行った場合
$ du -h models/baseline/*
  2.5G    biencoder.pt
  13G     embedding.pickle
```


### データセットの質問に関連する文書の抽出
データセットの質問に関連する文書を抽出します。質問エンコーダから取得した質問エンベッディングと文書エンベッディングに対して Faiss を用いて類似度を計算します。
- [retrievers/DPR/scripts/retriever/retrieve_passage_of_dev.sh](retrievers/DPR/scripts/retriever/retrieve_passage_of_dev.sh)

#### 設定

```bash
$ vim scripts/configs/config.pth
```

- データセットやモデルを任意の場所に保存した方は、上記設定ファイルに以下の項目を設定してください。
    - `WIKI_FILE`：Wikipedia の文書集合ファイル
    - `TRAIN_FILE`：第4回訓練データ（Retriever(=DPR)の訓練を行う場合に設定が必要な項目です）
    - `DEV_FILE`：第4回開発データ（問題のみ）
    - `TEST_FILE`：第4回リーダーボード用データ
    - `DIR_DPR`：モデルや文書エンベッディングが保存されているディレクトリへのパス
    - `DIR_RESULT`: 関連文書抽出結果の保存先

下記のコマンドを実行することで、DPRがデータセットの質問に関連する文書を抽出します。
なお、運営の実行環境（NVIDIA GeForce GTX 1080 Ti x3）では、第4回開発データに対する関連文書の抽出に1時間弱要しました。

```bash
# 実行例

$ exp_name="baseline"
$ model="models/baseline/biencoder.pt"
$ embed="models/baseline/embedding.pickle"
$ targets="dev"  # {train, dev, test} から関連文書抽出対象を「スペースなしの ',' 区切り」で指定してください

$ bash scripts/retriever/retrieve_passage.sh \
    -n $exp_name \
    -m $model \
    -e $embed \
    -t $targets

# 実行結果
$ ls ${DIR_RESULT}/${exp_name}/retrieved
    dev_aio_pt.json   # 予測結果
    dev_aio_pt.tsv    # 予測スコア（Acc@k を含む）
    logs/
      predict_aio_pt.log                               # 実行時ログ
```

<br>

## Reader (Fusion-in-Decoder)

Fusion-in-Decoder(FiD) は、質問と各関連文書を連結したものをエンコーダーでベクトル化し、それを連結したものをデコーダーに入力することで解答を生成するモデルです。


## データセット

### 作成

前節のRetrieverによる関連文書抽出結果を任意の場所に保存した方は、[/app/datasets.yml](datasets.yml) ファイルを編集して下さい。

```bash
$ cd /app
$ vim datasets.yml
```

このファイルには、Retriever(Dense Passage Retrieval) によって検索された関連文書と質問を含むファイルへのパスを、下記に合わせて設定して下さい。

```yml
DprRetrieved:
  path: JaqketAIO.load_jaqketaio2
  class: JaqketAIO
  data:
#   train: retrievers/DPR/result/baseline/retrieved/train_aio_pt.json
    dev: retrievers/DPR/result/baseline/retrieved/dev_aio_pt.json
#   test: retrievers/DPR/result/baseline/retrieved/test_aio_pt.json
```

Retriever で訓練データやリーダーボード用データに対する関連文書抽出を行った場合は，上記ファイル内で`train` や `test` に対応する項目も設定して下さい。

<hr>

設定が完了したら、次に Reader 用にデータセット形式を変換します。

```bash
$ python prepro/convert_dataset.py DprRetrieved fusion_in_decoder
```

変換後のデータセットは次のディレクトリに保存されます。

```yaml
/app/datasets/fusion_in_decoder/DprRetrieved/dev.jsonl
```

### 形式
以下のインスタンスからなる JSONL ファイルを使用します。
なお、評価データを用いる際は答えが含まれていないため、`target`は空文字列となります。

```json
{
    "id": "(str) 質問ID",
    "question": "(str) 質問",
    "target": "(str) answers から一つ選択した答え。ない場合はランダムに選択される。",
    "answers": "(List[str]) 答えのリスト",
    "ctxs": [{
        "id": "(int) 記事ID",
        "title": "(str) Wikipedia 記事タイトル",
        "text": "(str) Wikipedia 記事",
        "score": "(float) retriever の検索スコア (ない場合は 1/idx で置換される。generator では使用されない。)",
        "has_answer": "(bool) 'text'内に答えが含まれているかどうか"
    }]
}
```

## Reader による解答生成と評価
はじめに、下記ディレクトリに移動して下さい。
```bash
$ cd generators/fusion_in_decoder
```

### FiDモデルについて
- [学習済みモデルのダウンロード](#学習済みモデルのダウンロード)節で既に学習済みReader(FiD)のダウンロードを行った場合は、既にモデルの準備が完了しているため、この節はスキップしてください。
- Reader (Fusion-in-Decoder) の学習については、[generators/fusion_in_decoder/README.md](generators/fusion_in_decoder/README.md) を参照して下さい。

```bash
# 学習済みReader(FiD)のダウンロードを行った場合
$ du -h models_and_results/baseline/*
  4.0K       config.json
  1.7G       optimizer.pth.tar
  851M       pytorch_model.bin
```


### 解答生成と評価

#### 設定

```bash
$ vim configs/test_generator_slud.yml
```

- データセットなどを任意の場所に保存した方は、上記設定ファイルに以下の項目を設定して下さい。
    - `name`：生成される解答テキストファイルの保存先
    - `eval_data`：評価したい変換後のデータセットへのパス（=評価データ）
    - `checkpoint_dir`：`name`ディレクトリが作成されるディレクトリのパス（デフォルト：使用する Reader モデルが保存されているディレクトリ）
    - `model_path`：使用する Reader モデルが保存されているディレクトリへのパス
- デフォルトでは出力した解答候補の生成確率が 85.0% 以上であるときに、その候補を解答とする（＝早押しボタンを押すことに対応）仕様になっていますが、上記設定ファイルの以下の項目の値を変更することで、生成確率に対する閾値を変更することができます。
    - `threshold_probability`：早押しボタンを押すための、解答候補の生成確率に対する閾値

#### 解答生成

学習済み生成モデルにより、解答を生成します。<br>
下記スクリプトを実行することで、質問に対する解答をReaderが生成します。<br>
モデルが早押しボタンを押さないと判断した場合、解答として`null`が出力されます。
- [scripts/test_generator.sh](generators/fusion_in_decoder/scripts/test_generator.sh)

なお、運営の実行環境（NVIDIA GeForce GTX 1080 x1）では、第4回開発データに対する解答の生成に約5時間を要しました。

```bash
# 実行例
$ bash scripts/test_generator.sh configs/test_generator_slud.yml

# 実行結果
$ ls ${checkpoint_dir}/${name}
    final_output.jsonl         # 生成された解答が出力されたファイル
```

- 関連文書の上位 60 件の文書を用いた時の、第4回開発データに対する解答出力の例
```json lines
{"qid": "AIO04-0001", "position": 1, "prediction": null}
{"qid": "AIO04-0001", "position": 2, "prediction": null}
{"qid": "AIO04-0001", "position": 3, "prediction": null}
{"qid": "AIO04-0001", "position": 4, "prediction": null}
{"qid": "AIO04-0001", "position": 5, "prediction": "終戦"}
```

#### 評価

早押し設定に対応した評価スクリプト`compute_score.py`を実行し、評価を行います。

```bash
$ cd ../../
$ python compute_score.py \
      --prediction_file generators/fusion_in_decoder/${checkpoint_dir}/${name}/final_output.jsonl \
      --gold_file retrievers/DPR/datasets/aio4/aio_04_dev_v1.0.jsonl \
      --limit_num_wrong_answers 3
```

引数の説明は以下の通りです。
- `--limit_num_wrong_answers`：1つの問題に対して、この引数で渡した回数誤答した場合、その問題は得点なしとして計算されます。

評価スクリプトが正しく実行された場合、以下のような出力が得られます。

```bash
# 出力例
    num_questions: 500
    num_correct: 228
    num_missed: 253
    num_failed: 19
    accuracy: 45.6%
    accuracy_score: 228.000
    position_score: 75.151
    total_score: 303.151
```

出力の説明は以下のとおりです。
- `num_questions`：評価データの問題数
- `num_correct`：正解数
- `num_missed`：誤答は`--limit_num_wrong_answers`回未満だが、正解を出力できなかった問題数
- `num_failed`：誤答を`--limit_num_wrong_answers`回以上した問題数
- `accuracy`：正解率
- `accuracy_score`：正解数に対する得点（正解数×1.0点）
- `position_score`：早押しボタンを押した時点に応じた得点
- `total_score`：総合得点（`accuracy_score`と`position_score`の総和）

スコアの詳細については、第4回AI王 web サイトの早押し解答部門「スコアの算出方法」もご参照ください。



## 謝辞・ライセンス

- 学習データに含まれるクイズ問題の著作権は [abc/EQIDEN 実行委員会](http://abc-dive.com/questions/) に帰属します。東北大学において研究目的での再配布許諾を得ています。
- 開発データは クリエイティブ・コモンズ 表示 - 継承 4.0 国際 ライセンスの下に提供されています。
  - <img src="https://i.imgur.com/7HLJWMM.png" alt="" title="">
- 開発/評価用クイズ問題は [株式会社キュービック](http://www.qbik.co.jp/) および [クイズ法人カプリティオ](http://capriccio.tokyo/) へ依頼して作成されたものを使用しております。
