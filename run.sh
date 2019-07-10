#
# run.sh
#

python train.py --network VGG11 --batch-size 7 --epochs 2 --lr 0.001 --load 2 --data-folder ./network-VGG11-batch-5-epochs-1-lr-001-load-5

#python train.py --network ResNet18 --batch-size 5 --epochs 2 --lr 0.001 --load 5 --data-folder ./network-ResNet18-batch-5-epochs-1-lr-001-load-5

#python train.py --resnet-depth 18 --batch-size 20 --epochs 30 --lr 0.001 --load 2 --data-folder ./depth-18-batch-20-epochs-30-lr-001-load-2

#python train.py --resnet-depth 18 --batch-size 20 --epochs 30 --lr 0.001 --load 5 --data-folder ./depth-18-batch-20-epochs-30-lr-001-load-5

#python train.py --resnet-depth 34 --batch-size 20 --epochs 50 --lr 0.001 --load 2 --data-folder ./depth-34-batch-20-epochs-50-lr-001-load-2

#python train.py --resnet-depth 34 --batch-size 20 --epochs 50 --lr 0.001 --load 5 --data-folder ./depth-34-batch-20-epochs-50-lr-001-load-5
