
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <cstdint>
#include <chrono>

// length of one vector
#define L 1000
// number of 32-bit ints needed to store one vector
#define L_32 ((L + 31) / 32)
// number of vectors
#define N 100000

// some macro I created for generating semi-random vectors
#define seed(n) (n * n + 7)



// returns n-th bit of a
#define GET_BIT(a, n)   (uint32_t)(((a) & ((uint32_t)1 << (n))) >> (n))
// returns a copy of a with n-th bit set to 1
#define SET_BIT(a, n)   (uint32_t)((a) | (1 << (n)))
// returns a copy of a with n-th bit set to 0
#define CLEAR_BIT(a, n) (uint32_t)((a) & (~(1 << (n))))
// returns a copy of a with n-th bit flipped
#define FLIP_BIT(a, n)  (uint32_t)((a) ^ (1 << (n)))

// returns bit_no bit from bigger bit vector
#define get_bit(vector, bit_no)   (GET_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// sets bit_no bit in bigger bit vector to 1 (modifies it)
#define set_bit(vector, bit_no)   ((vector)[(bit_no) / 32] = SET_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// sets bit_no bit in bigger bit vector to 0 (modifies it)
#define clear_bit(vector, bit_no) ((vector)[(bit_no) / 32] = CLEAR_BIT((vector)[(bit_no) / 32], (bit_no) % 32))
// flips bit_no bit in bigger bit vector (modifies it)
#define flip_bit(vector, bit_no)  ((vector)[(bit_no) / 32] = FLIP_BIT((vector)[(bit_no) / 32], (bit_no) % 32))

// returns pointer to n-th vector in data
#define vector(data, n) (&(data)[(n) * L_32])


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

// this contains either two pointers to child nodes
// or two pointers to bit_vectors (leaf nodes)
// in either case, one of these pointers may be nullptr
struct node {
    union {
        struct node* children[2] = { nullptr, nullptr };
        uint32_t* bit_vector[2];
    };
};


// custom allocation in cuda memory
// as I need to preserve the whole tree structure
// and not just the underlying array
// I need have it in GPU-accessible memory
// and so I malloc a large block of memory
// big enough to fit the tree in the worst-case scenario
// and the vectors themselves
uint64_t mem_size = L * N;
uint64_t mem_current = 0;
struct node* memory;

struct node* alloc_node_in_memory()
{
    if (mem_current == mem_size)
    {
        // there could be a more graceful exit here
        // but from mathematical perspective, this will NEVER execute
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
};


// just to check if the tree works properly
// prints just the leaves from left to right
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
            // I created that empirically, seems to generate random enough output
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
            // in order to prevent duplicates from being printed
            // and to reduce the computation time +- by half
            // we can check just the vectors which have a 0 changed to 1
            // and still we will not miss a solution
            // so if currently inspected bit is a 1, we go to the next
            if (get_bit(vec, j))
                continue;

            // we flip j-th bit
            flip_bit(vec, j);

            // get a pointer to the tree root
			struct node* current = memory;
            // flag to indicate if we were able to go to the bottom
            bool searchFailed = false;
			for (int i = 0; i < L - 1; i++)
			{
                // traverse the tree
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


void solve_on_cpu(uint32_t* data, uint32_t* results, struct node* memory)
{
    for (int i = 0; i < N; i ++)
    {
        printf("Solving %d\n", i);
        // here we look for solutions for i-th vector

        uint32_t* res = vector(results, i);
        uint32_t* vec = vector(data, i);
        // iterate through its bits
        // to generate all possible vectors with Hamming distance 1
        // and for each check if it is in the initial input
        // and if it is, mark in the results vector
        for (int j = 0; j < L; j++)
        {
            // in order to prevent duplicates from being printed
            // and to reduce the computation time +- by half
            // we can check just the vectors which have a 0 changed to 1
            // and still we will not miss a solution
            // so if currently inspected bit is a 1, we go to the next
            if (get_bit(vec, j))
                continue;

            // we flip j-th bit
            flip_bit(vec, j);

            // get a pointer to the tree root
			struct node* current = memory;
            // flag to indicate if we were able to go to the bottom
            bool searchFailed = false;
			for (int i = 0; i < L - 1; i++)
			{
                // traverse the tree
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
    if (cudaStatus != cudaSuccess)
    {
        return;
    }

    uint32_t* results;
    cudaStatus = cudaMallocManaged((void**)&results, N * L_32 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto resultsError;
    }
    cudaStatus = cudaMemset((void*)results, 0, N * L_32 * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemsetfailed!");
        goto resultsError;
    }

    // allocate memory for the tree
    cudaStatus = cudaMallocManaged((void**)&memory, mem_size * sizeof(struct node));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed!");
        goto memoryError;
    }

    struct tree_custom_alloc* t = new struct tree_custom_alloc();

    // unfortunately the tree cannot be created in a parallel manner
    // so I construct it on cpu
    for (int i = 0; i < N; i++)
    {
        //printVectorAsString(vector(data, i));
        t->insert(vector(data, i));
    }

    //printTheTree(t->root, 0);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    solve<<<numBlocks, blockSize>>>(data, results, memory);

    cudaStatus = cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    solve_on_cpu(data, results, memory);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);

    printf("CPU time: %lld ns\n", cpu_time.count());
    printf("GPU time: %lld ns\n", gpu_time.count());

    // print the results
    // the form of results was not specified
    // so I just print them
    // the print coud be redirected to a file 
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

Error:
    cudaFree(memory);
memoryError:
    cudaFree(results);
resultsError:
    cudaFree(data);
    return 0;
}

