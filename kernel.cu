
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <bitset>


#define L 1000
#define L_32 ((L + 31) / 32)
#define N 100000
#define seed(n) (n * n + 7)


#include <cstdint>

// returs n-th bit of a
#define GET_BIT(a, n)   (uint32_t)(((a) & ((uint32_t)1 << (n))) >> (n))
// returns a copy of a with n-th bit set to 1
#define SET_BIT(a, n)   (uint32_t)((a) | (1 << (n)))
// returns a copy of a with n-th bit set to 0
#define CLEAR_BIT(a, n) (uint32_t)((a) & (~(1 << (n))))
// returns a copy of a with n-th bit flipped
#define FLIP_BIT(a, n)  (uint32_t)((a) ^ (1 << (n)))

// return bit_no bit from bigger bit vector
#define get_bit(vector, bit_no)   (GET_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// sets bit_no bit in bigger bit vector to 1 (modifies it)
#define set_bit(vector, bit_no)   ((vector)[(bit_no) / 32] = SET_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// sets bit_no bit in bigger bit vector to 0 (modifies it)
#define clear_bit(vector, bit_no) ((vector)[(bit_no) / 32] = CLEAR_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// flips bit_no bit in bigger bit vector (modifies it)
#define flip_bit(vector, bit_no)  ((vector)[(bit_no) / 32] = FLIP_BIT((vector)[(bit_no) / 32], (bit_no) % 32))

// return pointer to n-th vector in data
#define vector(data, n)         (&(data)[(n) * L_32])


bool hash(int a, int b)
{
    return ((a * seed(a) + b) % (a > b ? a - b : b - a)) % 2;
}

void printVectorAsString(uint32_t* v)
{
    std::string s(L, '0');
    for (int i = 0; i < L; i++)
    {
        if (get_bit(v, i) == 1)
        {
            s[i] = '1';
        }
    }
    printf("%s\n", s.c_str());
}

// the contains either two pointers to child nodes
// or two pointers to bit_vectors (leaf nodes)
// in either case, one of these pointers may be nullptr
struct node {
    union {
        struct node* children[2] = { nullptr, nullptr };
        uint32_t* bit_vector[2];
    };
};


// custom allocation in cuda memory


uint64_t mem_size = L * N;
uint64_t mem_current = 0;
struct node* memory;

struct node* alloc_node_in_memory()
{
    if (mem_current == mem_size)
    {
        abort();
    }
    struct node* tmp = memory + mem_current;
    mem_current += 1;
    tmp->children[0] = nullptr;
    tmp->children[1] = nullptr;
    return tmp;
}


// the tree structure I am using is basically a Trie
// which is a tree used for prefix matching
// in my case all words have length L
// and are from alphabet {0, 1}

struct tree_custom_alloc {
    struct node* root = alloc_node_in_memory();
    void insert(uint32_t* bit_vector)
    {
        // go down the tree, choosing direction based on current bit
        struct node* current = root;
        for (int i = 0; i < L - 1; i++)
        {
            if (current->children[get_bit(bit_vector, i)] == nullptr)
            {
                // if there is no child, create it
                current->children[get_bit(bit_vector, i)] = alloc_node_in_memory();
            }
			// and descend to it
			current = current->children[get_bit(bit_vector, i)];
        }
        if (current->bit_vector[get_bit(bit_vector, L - 1)] != nullptr)
        {
            // it the vector is there, we found a duplicate
            // so we don't care about it
            return;
        }
        // we add the vector in the appropriate place
        current->bit_vector[get_bit(bit_vector, L - 1)] = bit_vector;
    }
    bool exists(uint32_t* bit_vector)
    {
        // go down the tree, choosing direction based on current bit
        struct node* current = root;
        for (int i = 0; i < L - 1; i++)
        {
            if (current->children[get_bit(bit_vector, i)] != nullptr)
            {
                current = current->children[get_bit(bit_vector, i)];
            }
            else
            {
                // abort if there is no child at the specified location
                return false;
            }
        }
        // if we were able to go to the bottom
        // we are just above the leaves (vectors)
        // so we check if the requested vector exists
        if (current->bit_vector[get_bit(bit_vector, L - 1)] != nullptr)
        {
            return true;
        }
        return false;
    }
    void delete_tree(struct node* n, int depth)
    {
        // no need to delete anything in this case
        return;
    }
};

struct tree {
    struct node* root = new struct node();
    void insert(uint32_t* bit_vector)
    {
        // go down the tree, choosing direction based on current bit
        struct node* current = root;
        for (int i = 0; i < L - 1; i++)
        {
            if (current->children[get_bit(bit_vector, i)] == nullptr)
            {
                // if there is no child, create it
                current->children[get_bit(bit_vector, i)] = new struct node;
            }
			// and descend to it
			current = current->children[get_bit(bit_vector, i)];
        }
        if (current->bit_vector[get_bit(bit_vector, L - 1)] != nullptr)
        {
            // it the vector is there, we found a duplicate
            // so we don't care about it
            return;
        }
        // we add the vector in the appropriate place
        current->bit_vector[get_bit(bit_vector, L - 1)] = bit_vector;
    }
    bool exists(uint32_t* bit_vector)
    {
        // go down the tree, choosing direction based on current bit
        struct node* current = root;
        for (int i = 0; i < L - 1; i++)
        {
            if (current->children[get_bit(bit_vector, i)] != nullptr)
            {
                current = current->children[get_bit(bit_vector, i)];
            }
            else
            {
                // abort if there is no child at the specified location
                return false;
            }
        }
        // if we were able to go to the bottom
        // we are just above the leaves (vectors)
        // so we check if the requested vector exists
        if (current->bit_vector[get_bit(bit_vector, L - 1)] != nullptr)
        {
            return true;
        }
        return false;
    }
    void delete_tree(struct node* n, int depth)
    {
        if (depth == L - 1)
        {
            delete n;
            return;
        }
        if (n->children[0] != nullptr)
        {
			this->delete_tree(n->children[0], depth + 1);
        }
        if (n->children[1] != nullptr)
        {
            this->delete_tree(n->children[1], depth + 1);
        }
        delete n;
    }
};

struct arrays
{
    uint32_t* data;
    uint32_t* results;
    struct node* memory;
};

void printTheTree(struct node* n, int depth)
{
    if (depth == L - 1)
    {
        if (n->bit_vector[0] != nullptr)
        {
            printVectorAsString(n->bit_vector[0]);
        }
        if (n->bit_vector[1] != nullptr)
        {
            printVectorAsString(n->bit_vector[1]);
        }
    }
    else
    {
        if (n->children[0] != nullptr)
        {
            printTheTree(n->children[0], depth + 1);
        }
        if (n->children[1] != nullptr)
        {
            printTheTree(n->children[1], depth + 1);
        }
    }
}


// generating data is not part of the task itself
// so I do it on gpu in both versions of the algorithm
__global__
void generateDataKernel(uint32_t* data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        // generate one vector
        for (int j = 0; j < L; j++)
        {
            // set desired bit to 0 or 1
            // depending on the output of some function
            // data[i].set(j, hash(index, j));
			if (((index * seed(index) + j) % (index > j ? index - j : j - index) + 3) % 2)
			{
                set_bit(vector(data, i), j);
			}
            else
            {
                clear_bit(vector(data, i), j);
            }
        }
    }
}

__global__
void solve(uint32_t* data, uint32_t* results, struct node* memory)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        // here we look for solutions for i-th vector

        uint32_t* res = vector(results, i);
        uint32_t* vec = vector(data, i);
        // iterate through its bits
        // to generate all possible vectors with Hamming distance 1
        // and for each check if it is in the initial input
        // and if it is, mark in the results vector
        for (int j = 0; j < L; j++)
        {
            // !!! important: !!!
            // somehow copy vec value to tmp
            // probably best to do that not here
            // but when the data is created
            // so to the generation I shoud add just one line
            // so the modifications affect both vectors in tmp and data

            // we flip j-th bit
            flip_bit(vec, j);

			struct node* current = memory;
            bool searchFailed = false;
			for (int i = 0; i < L - 1; i++)
			{
				if (current->children[get_bit(vec, i)] != nullptr)
				{
					current = current->children[get_bit(vec, i)];
				}
				else
				{
					// abort if there is no child at the specified location
                    searchFailed = true;
				}
			}
            if (!searchFailed)
            {
				// if we were able to go to the bottom
				// we are just above the leaves (vectors)
				// so we check if the requested vector exists
				if (current->bit_vector[get_bit(vec, L - 1)] != nullptr)
				{
					// if it does, we flip the respective bit in results
                    flip_bit(res, j);
				}
            }
			// then we bring our vector back to its initial state
            flip_bit(vec, j);
        }
    }
}

cudaError_t allocAndGenerateData(uint32_t** data)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMallocManaged((void**)data, N * L_32 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto Error;
    }
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    generateDataKernel<<<numBlocks, blockSize>>>(*data);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }
Error:
    return cudaStatus;
}

int main()
{
    //get input somehow
    cudaError_t cudaStatus;
    uint32_t* data;
    cudaStatus = allocAndGenerateData(&data);
    //struct tree *t = new struct tree();

    // allocate memory for the results - the same as input data
    // but all bytes initialized to zero
    // if for n-th vector we detect that a vector with i-th bit flipped
    // is in the initial data, we indicate that by setting i-th bit
    // of n-th vector in this array to 1

    uint32_t* results;
    cudaStatus = cudaMallocManaged((void**)&results, N * L_32 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto Error;
    }
    cudaStatus = cudaMemset((void*)results, 0, N * L_32 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemsetfailed!");
        goto Error;
    }

    // allocate memory for the tree
    cudaStatus = cudaMallocManaged((void**)&memory, mem_size * sizeof(struct node));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto Error;
    }

    struct tree_custom_alloc* t = new struct tree_custom_alloc();

    for (int i = 0; i < N; i++)
    {
        //printVectorAsString(vector(data, i));
        t->insert(vector(data, i));
    }

    printTheTree(t->root, 0);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    solve<<<numBlocks, blockSize>>>(data, results, memory);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }

    // print the results
    for (int i = 0; i < N; i++)
    {
        uint32_t* vec = vector(data, i);
        uint32_t* res = vector(results, i);
        for (int j = 0; j < L; j++)
        {
            if (get_bit(res, j))
            {
				printf("\nPair:\n");
                printVectorAsString(vec);
                flip_bit(vec, j);
                printVectorAsString(vec);
                flip_bit(vec, j);
            }
        }
    }

    t->delete_tree(t->root, 0);
    cudaFree(memory);
    cudaFree(results);
Error:
    cudaFree(data);
    return 0;
}

