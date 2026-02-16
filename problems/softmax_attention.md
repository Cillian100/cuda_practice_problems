## SoftMax Attention

Implement a GPU program that computes the softmax attention operation for a given set of matrices.
Given the query matrix Q of size Mxd, key matrix K of size Nxd, and value matrix V of size Nxd, your
program should compute the output matrices using the formula.

Attention(Q,K,V)=softmax(QK^T/sqrt(d))V

where the softmax function is applied row-wise

