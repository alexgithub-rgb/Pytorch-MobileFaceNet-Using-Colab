# Pytorch-MobileFaceNet-Using-Colab
the code is almost from https://github.com/yeyupiaoling/Pytorch-MobileFaceNet.git ,while change some codes that we can use on colab

training dataset and test dataset can be downloaded on kaggle or https://github.com/yeyupiaoling/Pytorch-MobileFaceNet

create_dataset.py  see_model.py should run on colab cpu 

eval.py  train.py should run on colab gpu

add partial(10,25,50) to convert_data(root_path, output_prefix,partial) to choose the proportion of trainig data

add mobilefacenet_reduce.py to reduce 21% of parameters

run main.ipnb
change parameters in train.py and  the code in create_dataset.py and eval.py




