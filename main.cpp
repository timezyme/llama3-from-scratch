
#include "config.h"
#include "tokenizer.h"
#include <iomanip>
#include <iostream>

using namespace std;

const int MAX_BATCH_SIZE = 16;
const int MAX_SEQUENCE_SIZE = 64;

void show_tokenize(string input);

int main() {


    show_tokenize("hello world");

    return 0;
}

// if you implement tokenizer_bpe you should see token ids here. 
void show_tokenize(string input = "Hello world") {
    BPETokenizer tok(TOKENIZER_PATH);

    auto ids = tok.encode(input);
    for (int &id : ids) {
        cout << id << " ";
    }

    cout << "Token IDs: ";
    for (int id : ids) {
        cout << id << " ";
    }
    cout << "\n";
}
