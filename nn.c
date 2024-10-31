#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int CORPUS_SIZE = 160000;
int corpus[CORPUS_SIZE];
int tokens[CORPUS_SIZE];

const int VOCAB_SIZE = 96;
const char INT_TO_CHAR[96] = {
        '\n', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' '
};

int get_token_id(char token) {
    for (int i = 0; i < VOCAB_SIZE;  ++i) {
        if (token == INT_TO_CHAR[i])
            return i;
    }
    return 0;
}

const char BOS_TOKEN = '$', EOS_TOKEN = '@', PAD_TOKEN = '#';
int BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID;

const float lr = 1e-1;
const int MAX_STEPS = 1000;
const int B = 128;
const int EMBED_DIM = 64;
const int CONTEXT_LENGTH = 16;
const int HIDDEN_DIM = 64;
const int INPUT_DIM = CONTEXT_LENGTH * EMBED_DIM;

int input_tokens[B][CONTEXT_LENGTH];
int target_tokens[B];

// embeddings
float embeddings[VOCAB_SIZE][EMBED_DIM];
float embeddings_grad[VOCAB_SIZE][EMBED_DIM];

// input
float input[B][INPUT_DIM];
float input_grad[B][INPUT_DIM];

// hidden
float hidden[B][HIDDEN_DIM];
float hidden_grad[B][HIDDEN_DIM];
float hidden_act[B][HIDDEN_DIM];
float hidden_act_grad[B][HIDDEN_DIM];

// output
float output[B][VOCAB_SIZE];
float output_grad[B][VOCAB_SIZE];

// output probabilities
float q[B][VOCAB_SIZE];

// W1, b1
float W1[HIDDEN_DIM][INPUT_DIM];
float W1_grad[HIDDEN_DIM][INPUT_DIM];
float b1[HIDDEN_DIM];
float b1_grad[HIDDEN_DIM];

// W2, b2
float W2[VOCAB_SIZE][HIDDEN_DIM];
float W2_grad[VOCAB_SIZE][HIDDEN_DIM];
float b2[VOCAB_SIZE];
float b2_grad[VOCAB_SIZE];

// Function to generate Gaussian-distributed random numbers using Box-Muller transform
float random_normal(float mean, float std) {
    // Generate two uniform random numbers in the range (0, 1)
    double u1 = ((double) rand() + 1.0) / (RAND_MAX + 1.0);
    double u2 = ((double) rand() + 1.0) / (RAND_MAX + 1.0);

    // Apply Box-Muller transform
    float z0 = sqrtf(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

    // Scale and shift to the desired mean and standard deviation
    return z0 * std + mean;
}

void init_weight() {
    // use xavier init for stable training without batch norm
    float mean = 0.0f, std = 1.0f;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < EMBED_DIM; ++j) {
            embeddings[i][j] = random_normal(mean, sqrtf(2.0f / (VOCAB_SIZE + EMBED_DIM)));
        }
    }

    for (int i = 0; i < HIDDEN_DIM; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            W1[i][j] = random_normal(mean, sqrtf(2.0f / (INPUT_DIM + HIDDEN_DIM)));
        }
        b1[i] = 0;
    }
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            W2[i][j] = random_normal(mean, sqrtf(2.0f / (HIDDEN_DIM + VOCAB_SIZE)));
        }
        b2[i] = 0;
    }
}

float relu(float x) {
    if (x > 0)
        return x;
    return 0;
}

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

float sigmoid_grad(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// function for obtaining probabilities from logits
void softargmax() {
    for (int b = 0; b < B; ++b) {
        float max_output = output[b][0];
        for (int i = 1; i < VOCAB_SIZE; ++i) {
            if (output[b][i] > max_output) {
                max_output = output[b][i];
            }
        }

        // use the log sum exp 'trick'
        float log_sum_exp = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            log_sum_exp += expf(output[b][i] - max_output);
        }
        log_sum_exp = max_output + logf(log_sum_exp);
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            q[b][i] = expf(output[b][i] - log_sum_exp);
        }

//        float sum_exp = 0.0f; // to accumulate the sum of exponentials
//        for (int i = 0; i < VOCAB_SIZE; ++i) {
//            q[b][i] = expf(output[b][i] - max_output); // compute exp
//            sum_exp += q[b][i]; // accumulate the sum
//        }
//
//        // Normalize to get probabilities
//        for (int i = 0; i < VOCAB_SIZE; ++i) {
//            q[b][i] /= sum_exp; // divide by the sum of exponentials
//        }
    }
}

float cross_entropy() {
    float loss = 0;
    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < CONTEXT_LENGTH; ++i) {
        loss += -logf(q[b][target_tokens[b]]);
//            printf("%f ", q[b][target_tokens[i]]);
//        }
//        printf("\n");
    }
    return loss / B;
}

void forward() {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < CONTEXT_LENGTH; ++i) {
            for (int j = 0; j < EMBED_DIM; ++j)
                input[b][EMBED_DIM * i + j] = embeddings[input_tokens[b][i]][j];
        }
    }
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < CONTEXT_LENGTH * EMBED_DIM; ++i) {
//            printf("%f ", input[b][i]);
//        }
//        printf("\n");
//    }
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            // reset hidden activations
            hidden[b][i] = 0;
            for (int j = 0; j < INPUT_DIM; ++j)
                hidden[b][i] += W1[i][j] * input[b][j];
            hidden[b][i] += b1[i];

//            printf("%f ", hidden[b][i]);

            // apply activation function
            hidden_act[b][i] = sigmoid(hidden[b][i]);
        }
//        printf("\n");
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            // reset output activations
            output[b][i] = 0;
            for (int j = 0; j < HIDDEN_DIM; ++j)
                output[b][i] += W2[i][j] * hidden_act[b][j];
            output[b][i] += b2[i];
//            printf("%f ", output[b][i]);
        }
//        printf("\n");
    }
    softargmax();
}

void backwards() {
//    float target[B][VOCAB_SIZE];
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < VOCAB_SIZE; ++i) {
//            if (target_tokens[b] == i)
//                target[b][i] = 1.0f;
//            else
//                target[b][i] = 0.0f;
//        }
//    }
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            float target = 0.0f;
            if (target_tokens[b] == i)
                target = 1.0f;
            assert(output_grad[b][i]);
            output_grad[b][i] = 1.0f / B * (q[b][i] - target);
        }
    }

    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            assert(W2_grad[i][j] == 0);
            for (int b = 0; b < B; ++b) {
                W2_grad[i][j] += output_grad[b][i] * hidden[b][j];
            }
        }
        assert(b2_grad[i] == 0);
        for (int b = 0; b < B; ++b) {
            b2_grad[i] += output_grad[b][i];
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            assert(hidden_act_grad[b][i] == 0);
            for (int j = 0; j < VOCAB_SIZE; ++j) {
//                if (hidden_relu[b][i] <= 0)
//                    break;
//              h = W1 @ x + b
//              h_act = sigmoid(h(x))
//              h_grad(x) = h_act_grad(h(x)) * h_act_grad_h(x)
//              h_act_grad_h = sigmoid(h(x)) * h(x)
                hidden_act_grad[b][i] += output_grad[b][j] * W2[j][i];
            }
            assert(hidden_grad[b][i] == 0);
            hidden_grad[b][i] = hidden_act_grad[b][i] * sigmoid_grad(hidden[b][i]);
//            printf("%f ", hidden_grad[b][i]);
        }
//        printf("\n");
    }

    for (int i = 0; i < HIDDEN_DIM; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            assert(W1_grad[i][j] == 0);
            for (int b = 0; b < B; ++b) {
                W1_grad[i][j] += hidden_grad[b][i] * input[b][j];
            }
        }
        assert(b1_grad[i] == 0);
        for (int b = 0; b < B; ++b) {
            b1_grad[i] += hidden_grad[b][i];
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < INPUT_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                input_grad[b][i] += hidden_grad[b][j] * W1[j][i];
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < CONTEXT_LENGTH; ++i) {
            for (int j = 0; j < EMBED_DIM; ++j) {
                embeddings_grad[input_tokens[b][i]][j] += input_grad[b][EMBED_DIM * i + j];
            }
        }
    }
//    for (int i = 0; i < VOCAB_SIZE; ++i) {
//        for (int j = 0; j < EMBED_DIM; ++j)
//            printf("%f ", embeddings_grad[i][j]);
//        printf("\n");
//    }
}

void update() {
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < EMBED_DIM; ++j)
            embeddings[i][j] -= lr * embeddings_grad[i][j];
    }

    for (int i = 0; i < HIDDEN_DIM; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j)
            W1[i][j] -= lr * W1_grad[i][j];
        b1[i] -= lr * b1_grad[i];
    }

    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j)
            W2[i][j] -= lr * W2_grad[i][j];
        b2[i] -= lr * b2_grad[i];
    }
}

void zero_grad() {
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < EMBED_DIM; ++j)
            embeddings_grad[i][j] = 0;
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < INPUT_DIM; ++i)
            input_grad[b][i] = 0;
    }

    for (int i = 0; i < HIDDEN_DIM; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j)
            W1_grad[i][j] = 0;
        b1_grad[i] = 0;
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            hidden_grad[b][i] = 0;
            hidden_act_grad[b][i] = 0;
        }
    }

    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j)
            W2_grad[i][j] = 0;
        b2_grad[i] = 0;
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < VOCAB_SIZE; ++i)
            output_grad[b][i] = 0;
    }
}

//void prepare_tokens(int start_index) {
//    int shift_by = CONTEXT_LENGTH + 1;
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < CONTEXT_LENGTH; ++i) {
//            input_tokens[b][i] = tokens[b * shift_by + start_index + i];
//        }
//        target_tokens[b] = tokens[b * shift_by + start_index + CONTEXT_LENGTH];
//    }
//}

void tokenize() {
    int index = 0;
    for (int i = 0; i < CORPUS_SIZE; ++i) {
        for (int j = 0; j < VOCAB_SIZE; ++j) {
//            printf("%c == %c, ", INT_TO_CHAR[j], corpus[i]);
            if (INT_TO_CHAR[j] == (char) corpus[i]) {
                tokens[index] = j;
                index++;
                break;
            }
        }
    }
//    for (int i = 0; i < CORPUS_SIZE; ++i) {
//        printf("%d", tokens[i]);
//        printf("%c, ", INT_TO_CHAR[tokens[i]]);
//    }
//    printf("\n");
}

void train() {
    int min = 0, max = CORPUS_SIZE - B * CONTEXT_LENGTH;
    int step = 1;
    FILE *log_file = fopen("nn.c.log", "w");
    while (step < MAX_STEPS) {
        // prepare a batch
        // int random_indices[B] = {0, 25, 50, 75, 100};
        int random_indices[B];
        for (int i = 0; i < B; ++i)
            random_indices[i] = min + rand() % (max - min + 1);
        
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < CONTEXT_LENGTH; ++j) {
                input_tokens[i][j] = tokens[random_indices[i] + j];
                // printf("%d ", input_tokens[i][j]);
            }
            // printf("\n");
        }
        for (int i = 0; i < B; ++i) {
            target_tokens[i] = tokens[random_indices[i] + CONTEXT_LENGTH];
            // printf("%d ", target_tokens[i]);
        }

        forward();
//        printf("step: %d\n", step);
        float loss = cross_entropy();
        fprintf(log_file, "step: %d, loss: %f\n", step, loss);
        printf("step: %d, loss: %f\n", step, loss);
        backwards();
        update();
        zero_grad();
        step++;
        // break;
    }
    fclose(log_file);
}

int main(int argc, char *argv[]) {
//    srand(time(NULL));
    srand(42);

    // Open the file in "read" mode
    FILE *file = fopen("dataset.txt", "r");

    // Check if the file was opened successfully
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return 1;
    }

    printf("Reading the file dataset.txt\n");
    int i = 0;
    while ((corpus[i] = fgetc(file)) != EOF) {
        if (i >= CORPUS_SIZE) break;
        i++;
    }
    fclose(file);

    BOS_TOKEN_ID = get_token_id(BOS_TOKEN);
    EOS_TOKEN_ID = get_token_id(EOS_TOKEN);
    PAD_TOKEN_ID = get_token_id(PAD_TOKEN);
    printf("BOS_TOKEN: %c (%d), EOS_TOKEN: %c (%d), PAD_TOKEN: %c (%d)\n",
           BOS_TOKEN, BOS_TOKEN_ID, EOS_TOKEN, EOS_TOKEN_ID,
           PAD_TOKEN, PAD_TOKEN_ID);
    tokenize();

    init_weight();
//    forward();
//    backwards();
    time_t start, end;
    start = time(NULL);
    train();
    end = time(NULL);

    // Calculate the difference in seconds
    double elapsed = difftime(end, start);
    printf("Time taken: %.2f seconds\n", elapsed);
    return 0;
}