# Implementation of KNN and K-Means

This program contains implementations of K-Means and K-NN algorithms, and can be used to 
train and test these models on some arbitrary data (in csv format). 

## How to Run

To run this program, simply use python3 as shown below. 
```
python3 main.py -h
```

The option `-h` can be used to get more information about the flags that can be used and 
the files that are required depending on the mode.

Flags that can be used are:

* `-mode` : the algorithm that will be used. It can be `knn` or `kmeans`
* `-k` : k value to use. This means the number of clusters. the default is 3.
* `-d` : Distance function to use. The possible values are: `e2` which represents the euclidean squared distance. And `manh` which represents the manhattan distance.
* `-unitw`: If present, unit voting weights will be used, if not, then 1/d weights will be used. 
* `-train`: a file containing training data. This file will only be used for KNN and ignored otherwise.
* `-test`: a file containing points to test. This file will only be used for KNN and ignored otherwise.
* `-data`: a file containing points. It will be used only for KMeans. If the mode is not KMeans, it will be ignored
* If the mode is `kmeans` then centroid need to be provided as positional arguments (arguments without flags)

## Running KNN

To run the program on some input using the KNN algorithm, you just need to set the mode to knn, 
specify a value for k, and pass the training and testing sets as input to the CLI.
```
python3 main.py -mode knn -k 3 -train ./training_file -test ./testing_file
```

## Running KMeans

To run the program on some input using the KMeans algorithm, you just need to set the mode to kmeans, pass the dataset as input to the CLI using the `-data` flag and specify any number of centroids.
```
python3 main.py -mode kmeans -data ./datafile 0,0 200,200 500,500
```

K-Means requires the initial centroids, this must be passed as command line arguments where each centroid is separated by a space and each coordinate is separated by a comma. 

## In Case of Ambiguity
This section explains the deterministic nature of the code. We could break ties using randomness, but,
it was easier to just be deterministic. 

Sometimes, if we are using KNN, the input files will contain points that will cause some kind of ambiguity. 
Since we are using python's sort method in KNN, and the sort method is stable, the output depends on the order 
in which the training points are given. 

We are also using a Counter object from the collections module to count the number of votes for each label. 
This also depends on the order in which the values are added to the counter, so this does not change the fact 
that we are using the order of the input files to break the ambiguity. 

A form of ambiguity that was still present in the sample files is that given some value of k, some instance 
that we are trying to label will be equidistant to the kth and k+1th neighbor, and if we were to swap these 
two elements (we can do that since they are at the same distance) the label would change if the kth element 
was the tiebreaker between the top 2 labels. This happens in the files knn3.train and knn3.test when the k 
value is 7, the manh distance function is being used, and we are using unit voting. I decided to display a 
message showing an explanation and the alternative label. Obviously, the recall and precision will be different, 
but it should still be correct.

To enable or disable the message saying that the label could have been different, use the flag `-alt`. For more information, use the flag `-h`