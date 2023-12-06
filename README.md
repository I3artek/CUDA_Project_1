# CUDA_Project_1

The algorithm I made:
- generates N semi-random vectors - not part of the task itself
- creates a tree from the vectors in a following manner:
  all vectors are at leaf nodes and each vector's value represents the path to get to it from the root of the tree
  so we start a tree root
  then we take a vector and iterate through its bits
  if a bit is 0, we go to the left (first child) and if it's 1, we go to the right (second child)
  while doing that, we create all the necessary intermediate nodes
  we do that for every vector and thus create a tree containing all the vectors
  since we go one level down for each bit of a vector, the tree has height L (L is the length of one vector)
  so, this means that travelling to any leaf node takes O(L) time, which is great
  and as to the tree creation, we iterate through L bits of N vectors, so the creation takes O(NL) time
- after that, we have our data representation ready, so we look for pairs of vectors with Hamming distance equal to 1
  but instead of analyzing every possible pair, for each vector we do the following:
  generate all the possible vectors that have Hamming distance equal to 1 when compared to the current vector
  this takes O(L) time, as we just need to flip every bit (actually we can just flip 1s to 0s or the other way)
  then, for each such vector, we check if it was in the original list of vector
  thanks to our tree, we can do that in O(L) time
  so, we do that for N vectors, resulting in complexity O(NL^2), wchich is a nice result in the end
  
From my checks, it seems the execution on the GPU is approximately 20 times faster than on cpu.
