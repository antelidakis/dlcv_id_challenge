#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -eq 1 ]; 
    then echo "Training"
	AnnotCsv="../data/gicsd_labels.csv"
	OutDir="../artifacts/idvggnet_upsampling_test/"
	echo $AnnotCsv
	echo $OutDir
	
	# Quick demo
	#python train.py --annot "${AnnotCsv}" --output "${OutDir}" --epochs 2
	
	# training with upsambling
	python train.py --annot "${AnnotCsv}" --output "${OutDir}" --upsampling 1
	
	# Training for balanced datasets
	# python train.py --annot "${AnnotCsv}" --output "${OutDir}" 
elif [ $# -eq 2 ]; 
    then echo "Prediction"
    ImagePath=$2
    PathModel="../artifacts/idvggnet_upsampling"
    PathLabels=${PathModel}/idvggnet_lb.pickle
    echo $ImagePath
    echo $PathModel
    echo $PathLabels
    python predict.py --image "${ImagePath}" --artifact "${PathModel}" --labels "${PathLabels}" 
else
	echo "Bad use. Try:"
	echo "./main.sh -train"
    echo "./main.sh -predict {path_to_image}"
fi