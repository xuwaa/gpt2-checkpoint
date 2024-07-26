**Status:** Archive (code is provided as-is, no updates expected)

# gpt-2



gpt2的dataset需要去 [released a dataset](https://github.com/openai/gpt-2-output-dataset) 下载 data1 data2 data3 data4.

python download_dataset.py 117M
python download_dataset.py 345M
python download_dataset.py 762M
python download_dataset.py 1542



## Usage

GPT-2模型文件. 查看[model card](./model_card.md).运行

python download_model.py 124M
python download_model.py 345M
python download_model.py 774M
python download_model.py 1558M

##Run

观察全体模型网络层运行结果

python src/train.py --dataset data4 --model_name 1558M --models_dir models --part all

观察部分模型网络层运行结果

python src/train.py --dataset data4 --model_name 1558M --models_dir models --part part

可以修改为其他例如，

python src/train.py --dataset data3 --model_name 774M --models_dir models --part part

要想对比某一时刻的两种重启差异，需要将all状态运行到指定epoch，例如counter==4001,当保存好当前的checkpoint文件，
run_name和run_name2分别对应all和part的checkpoint文件，只需要--part指定类型即可运行对应重启结果

项目设置了3h的运行时间。
