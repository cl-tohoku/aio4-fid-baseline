import os
import argparse
import json
import jsonlines
from tqdm import tqdm


def jsonl_to_list_of_dicts(jsonl_file_path):
    li = []
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            li.append(obj)
    return li


def main():
    """
    Usage: python slice_questions.py -n 10  # 10文字目まで質問文をスライスしたデータセットを作成
    """
    parser = argparse.ArgumentParser(description='文字数ごとにスライスされた問題文のデータセットを作成します。')

    parser.add_argument('--dataset_path', help='スライス対象のデータセットのパス', type=str, default="datasets/aio_02_dev_v1.0.jsonl")
    parser.add_argument('-n', '--slice_num', help='何文字目までスライスするか', type=int, required=True)
    parser.add_argument('--output_directory_path', help='スライスされたデータセットの保存先ディレクトリ', type=str, default="datasets/sliced_dataset")

    args = parser.parse_args()

    # データセットの読み取り
    list_of_dataset = jsonl_to_list_of_dicts(args.dataset_path)

    # n文字までスライスをする
    list_of_sliced_dataset = []
    letters_of_longest_question = 0
    for data in tqdm(list_of_dataset):
        if len(data["question"]) > letters_of_longest_question:
            letters_of_longest_question = len(data["question"])
        sliced_question = data["question"][:args.slice_num]
        data["question"] = sliced_question
        list_of_sliced_dataset.append(data)
    print(f"Longest question has {letters_of_longest_question} letters.")

    # スライスされたデータセットの保存
    os.makedirs(args.output_directory_path, exist_ok=True)
    sliced_dataset_name = str(args.slice_num) + "letters_" + args.dataset_path.split("/")[-1]
    save_path = os.path.join(args.output_directory_path, sliced_dataset_name)
    with open(save_path, mode="w", encoding="utf-8") as fout:
        for obj in list_of_sliced_dataset:
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    main()
