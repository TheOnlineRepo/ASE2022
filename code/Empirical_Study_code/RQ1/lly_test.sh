#! /bin/bash
#for model in "vgg13" "vgg16" "vgg19"
#do
#	for nc in "0.1" "0.25" "0.5" "0.75" "0.9"
#	do
#		python -u jaccard.py -dataset cifar100 -model $model -fitness nc -k_nc $nc
#	done
#
#	for tknc in 1 2 3 4 5
#	do
#		python -u jaccard.py -dataset cifar100 -model $model -fitness tknc -k_tknc $tknc
#	done
#	
#	for kmnc in 10 50 100 200
#	do
#		python -u jaccard.py -dataset cifar100 -model $model -fitness kmnc -k_kmnc $kmnc
#	done
#done
#
#for model in "WRN" "DenseNet121"
#for model in "vgg19"
#do
#	for nc in "0.1" "0.25" "0.5" "0.75" "0.9"
#	do
#		python -u jaccard.py -dataset SVHN -model $model -fitness nc -k_nc $nc
#	done
#
#	for tknc in 1 2 3 4 5
#	do
#		python -u jaccard.py -dataset SVHN -model $model -fitness tknc -k_tknc $tknc
#	done
#
#	for kmnc in 10 50 100 200
#	do
#		python -u jaccard.py -dataset SVHN -model $model -fitness kmnc -k_kmnc $kmnc
#	done
#done
for model in "vgg13" "vgg16" "vgg19"
#for model in "vgg13"
#for model in "WRN" "resnet34" "DenseNet121" "vgg19"
#for model in "vgg19"
#for model in "WRN"
#for model in "resnet34"
do
	for lsa in 10 50 100 200 500 1000
	do
		python -u jaccard.py -dataset cifar100 -model $model -fitness lsa -sa_n $lsa
		#python -u jaccard.py -dataset SVHN -model $model -fitness lsa -sa_n $lsa
	done

	#for dsa in 10 50 100 200 500 1000
	#do
	#	python -u jaccard.py -dataset SVHN -model $model -fitness dsa -sa_n $dsa
	#done
	#for idc in 4 6 8 10 12
	#do
	#	python -u jaccard.py -dataset SVHN -model $model -fitness idc -idc_n $idc
	#done
	#python -u jaccard.py -dataset SVHN -model $model -fitness lsa -sa_n 10
	#python -u jaccard.py -dataset SVHN -model $model -fitness dsa -sa_n 10
done

