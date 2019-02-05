# Properties_of_Embeddings
Properties of Vector Embeddings in Social Networks

# Datasets
[Facebook](https://snap.stanford.edu/data/egonets-Facebook.html) <br />
[Twitter](https://snap.stanford.edu/data/egonets-Twitter.html) <br />
[Google+](https://snap.stanford.edu/data/egonets-Gplus.html) <br />



# Paper
[Properties of Vector Embeddings in Social Networks](https://www.mdpi.com/1999-4893/10/4/109/pdf-vor)


# How to run
1) Learn embeddings using [Deepwalk](https://github.com/phanein/deepwalk), [node2vec](https://github.com/aditya-grover/node2vec) or [HARP](https://github.com/GTmac/HARP) <br />
2) Give the ranking label to every pair of nodes by prepare_labels.py  <br />
3) Run RankSVM.py to learn weights for each graph property <br />

