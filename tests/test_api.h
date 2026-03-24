// DO NOT CHANGE THIS FILE, Implement the functions in test_api.cpp
// If you have any signature issues, let us know.

#include "prelude.h"

class TestAPI {
  public:
    // dont forget padding with bos token
    vector<int> tokenize(string input);

    string detokenize(vector<int> token_ids);

    // If you do bf16 or fp16 dumping, you will need to convert the embeddings
    // to fp32 before returning them.
    vector<float> get_embeddings(vector<int> token_ids);

    // Matrix multiply (row-major): A[M,K] * B[K,N] -> C[M,N].
    // Inputs are flattened row-major buffers of sizes M*K and K*N.
    // Returns a flattened row-major vector of size M*N.
    // Everthing flat!
    vector<float> matmul(const vector<float> &A, const vector<float> &B, int M,
                         int K, int N);

    /*
    ***************  End of Milestone 1 API, rest of the functions for
    * Milestone-2 will be released later ***************
    */
};
