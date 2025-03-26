# MNIST
python generate_subsampled_data.py --alpha 0.0 --dataset_name MNIST_uniform --data_source MNIST
python generate_subsampled_data.py --alpha 0.2 --dataset_name MNIST_moderately_skewed --data_source MNIST
python generate_subsampled_data.py --alpha 1.0 --dataset_name MNIST_heavily_skewed --data_source MNIST
python generate_subsampled_data.py --alpha -1  --dataset_name MNIST_extremely_skewed --data_source MNIST

# CIFAR10
python generate_subsampled_data.py --alpha 0.0 --dataset_name CIFAR10_uniform --data_source CIFAR10
python generate_subsampled_data.py --alpha 0.2 --dataset_name CIFAR10_moderately_skewed --data_source CIFAR10
python generate_subsampled_data.py --alpha 1.0 --dataset_name CIFAR10_heavily_skewed --data_source CIFAR10
python generate_subsampled_data.py --alpha -1  --dataset_name CIFAR10_extremely_skewed --data_source CIFAR10

# FashionMNIST
python generate_subsampled_data.py --alpha 0.0 --dataset_name FashionMNIST_uniform --data_source FashionMNIST
python generate_subsampled_data.py --alpha 0.2 --dataset_name FashionMNIST_moderately_skewed --data_source FashionMNIST
python generate_subsampled_data.py --alpha 1.0 --dataset_name FashionMNIST_heavily_skewed --data_source FashionMNIST
python generate_subsampled_data.py --alpha -1  --dataset_name FashionMNIST_extremely_skewed --data_source FashionMNIST

