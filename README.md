**Status:** Archive (code is provided as-is, no updates expected)

# gpt-2



Gpt-2 dataset need to [released a dataset] (https://github.com/openai/gpt-2-output-dataset) to download data1 data2 data3 data4.

python download_dataset.py 117M

python download_dataset.py 345M

python download_dataset.py 762M

python download_dataset.py 1542



## Usage


GPT-2 model file. View [model card](./model_card.md). operation


python download_model.py 124M

python download_model.py 345M

python download_model.py 774M

python download_model.py 1558M

##Run

Observe the operation results of all layers in the model network. 


python src/train.py --dataset data4 --model_name 1558M --models_dir models --part all


Observe the results of running some of the model's network layers. 


python src/train.py --dataset data4 --model_name 1558M --models_dir models --part part


It can be modified to other examples, for instance.

python src/train.py --dataset data3 --model_name 774M --models_dir models --part part

To compare the differences between two restarts at a specific point in time, you need to run the all state to a specified epoch, such as counter == 4001. When you save the current checkpoint file, you can run the corresponding restart results by specifying the --part option and using run_name and run_name2 to correspond to the checkpoint files for all and part, respectively.

The project is set to run for 3 hours.

We complete using local memory to store checkpoints by running the move.py script.

python move.py
