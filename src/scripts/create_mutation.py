import numpy as np
import sys, os
import random
sys.path.append(os.path.abspath("/home/gourab/Desktop/Projects/Covid/Tempel-modified/source_code"))

from src.models import predict_model, our_models, lstm_simplified
import torch
from src.utils import utils
import json
from collections import OrderedDict
from itertools import product
from tqdm import tqdm
import pandas as pd
import itertools
import random
import numpy as np



def create_mutated_sequences():

    random.seed(100)

    dataset = 'data/independent_set/cov_test.csv'
    data_path = 'data/independent_set/protVec_100d_3grams.csv'

    test_trigram_vecs, test_labels = utils.read_dataset(dataset, data_path, concat=False)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)

    input_dim = X_test.shape[2]
    seq_length = X_test.shape[0]
    output_dim = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    net = lstm_simplified.LSTMSeriesClassifier5(
                seq_length=seq_length,
                input_size=input_dim,
                hidden_size=128,
                num_layers=2,
                num_classes=output_dim,
                dropout = 0.5,
                positional_dropout = 0.1,
                device=device
            )
               
    # # net = models.AttentionModel(seq_length, input_dim, output_dim, 128, 0.5)


    # # X_test.permute((1, 0, 2))

    labels, prec, rec, f, mcc, acc, _, _, _, _, _ = predict_model.test_model(
        model= net,
        model_type='our model',
        x_test=X_test,
        y_test=Y_test,
        device=device
    )

    print(acc, prec, rec, f, mcc)
    # # # # exit()

    with open('data/mutation_maps/class_0.json', 'r') as f:
         mutation = OrderedDict(json.load(f))

    sequences, prev_labels = [], []

    with open('data/independent_set/sequences.txt', 'r') as f:
        for _, line in enumerate(f):
            sequences.append(line)

    with open('data/independent_set/previous_labels.txt', 'r') as f:
        for _, line in enumerate(f):
            prev_labels.append(int(line))

    # np.random.seed(10)
    # random.seed(10)

    # print(len(sequences))

    # sequences = [
    #     'ABCDEFG',
    #     'BBCDEAG'
    # ]

    # print(sequences)

    # mutation = {
    #     2 : 'BC',
    #     3 : 'DE',
    #     4: 'A'
    # }

    # print(sequences)

    # labels = np.ones(6)
    # labels[1:3] = 0
    # # prev_labels = [1]
    # np.random.shuffle(labels)

    mutation_pos = list(mutation.keys())
    mutation_pos = [int(pos) for pos in mutation_pos]

    epitope_sites = len(mutation_pos)
    print(epitope_sites)

    print(labels)
    print(mutation.keys())
    print(mutation_pos)
    
    # for idx, sequence in enumerate(sequences):
    #     final_string = sequence
    #     for label, pos, pos_int in zip(labels[idx * epitope_sites: (idx+1) * epitope_sites], mutation_pos, mutation_pos_ints):
    #         if label == 1:
    #             final_string = final_string[:pos_int-1] + random.choice(mutation[pos]) + final_string[pos_int:]
    #     sequences[idx] = final_string

    # print(len(sequences))

    # output_path = "data/independent_set/mutation.csv"

    # with open(output_path, "w") as out:
    #     out.write("Original Sequence,Mutated Seqeunce,Previous Label\n")

    # valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # labels = torch.Tensor(labels)
    # labels = torch.reshape(labels, (len(sequences), -1))

    original_sequences = []
    mutated_sequences = []

    # for sequence, seq_labels, prev_label in tqdm(zip(sequences, labels, prev_labels)):
    #     mutation_sites = [int(mutation_pos[i]) for i, label in enumerate(seq_labels) if label == 1]
    #     if len(mutation_sites) > 0:
    #         print(f"{len(mutation_sites)}")
    #         for combination in product(valid_amino_acids, repeat=len(mutation_sites)):
    #             mutated_sequence = sequence[:]
    #             for aa, pos in zip(combination, mutation_sites):
    #                 if aa != mutated_sequence[pos]:
    #                     mutated_sequence = mutated_sequence[:pos] + aa + mutated_sequence[pos+1:]
    #                 else:
    #                     break
    #             else:
    #                 print(sequence)
    #                 print(mutated_sequence)
    #                 with open(output_path, "a") as out:
    #                     out.write(f"{sequence},{mutated_sequence},{prev_label}\n")


    output_path = "data/independent_set/mutation.csv"

    # with open(output_path, "w") as out:
    #     out.write("Original Sequence,Mutated Seqeunce,Previous Label\n")

    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    labels = torch.Tensor(labels)
    labels = torch.reshape(labels, (len(sequences), -1))

    def generate_sequences(sequences, mutation_matrix, mutation_sites, valid_replacements, max_sequences=200):
        new_sequences, original_sequences = [], []
        for i, sequence in tqdm(enumerate(sequences), total=1000, leave=False):
            counter = 0
            mutations = mutation_matrix[i]
            mutated_positions = [site for label, site in zip(mutations, mutation_sites) if label == 1]
            if mutated_positions:
                rejection_probability = 0.8
                progress_bar = tqdm(total=max_sequences, desc=f'Original Sequence {i}')
                while counter < max_sequences:
                    replacements = random.choices(valid_replacements, k=len(mutated_positions))
                    new_sequence = sequence[:]
                    for position, replacement_aa in zip(mutated_positions, replacements):
                        new_sequence = new_sequence[:position] + replacement_aa + new_sequence[position + 1:]
                    if any(new_sequence[pos] == sequence[pos] for pos in mutated_positions):
                        continue
                    if np.random.uniform() > rejection_probability:
                        new_sequences.append(new_sequence.rstrip())
                        original_sequences.append(sequence.rstrip())
                        counter += 1
                        progress_bar.update(1)
                        rejection_probability = 0.8  # Reset rejection probability to 0.8 when sequence is accepted
                    else:
                        rejection_probability *= 0.95  # Reduce rejection probability
                        if rejection_probability < 0.1:
                            rejection_probability = 0.8  # Reset rejection probability to 0.8
                progress_bar.close()
                if counter < max_sequences:
                    print(f'Warning: Could not generate {max_sequences} sequences for original sequence {i}.')
        return original_sequences, new_sequences



    original_sequences, mutated_sequences = \
        generate_sequences(sequences, labels, mutation_pos, valid_amino_acids, 200)


    # for sequence, seq_labels, prev_label in tqdm(zip(sequences, labels, prev_labels)):
    #     mutation_sites = [int(mutation_pos[i]) for i, label in enumerate(seq_labels) if label == 1]
    #     if len(mutation_sites) > 0:
    #         for i in range(len(valid_amino_acids)):
    #             for j in range(len(valid_amino_acids)):
    #                 if i != j:
    #                     mutated_sequence = list(sequence)
    #                     for pos in mutation_sites:
    #                         mutated_sequence[pos] = valid_amino_acids[i] if mutated_sequence[pos] == valid_amino_acids[j] else mutated_sequence[pos]
    #                     mutated_sequence = "".join(mutated_sequence)
    #                     original_sequences.append(sequence.rstrip())
    #                     mutated_sequences.append(mutated_sequence.rstrip())
                        # with open(output_path, "a") as out:
                        #     out.write(f"{sequence.rstrip()},{mutated_sequence.rstrip()},{prev_label}\n")


    # print(sequences)

    print(len(mutated_sequences))

    # counts = []

    # for seq1 in mutated_sequences:
    #     count = sum(1 for a, b in zip(seq1, sequences[0]) if a != b)
    #     counts.append(count)
    

    # print(len(mutated_sequences))

    seqs = pd.DataFrame({
        "Original_sequence" : original_sequences,
        "Mutated_sequences" : mutated_sequences
    })

    print(seqs.shape)

    seqs.to_csv(output_path, index=False)


    # with open('data/independent_set/mutated_sequences.txt', 'w') as f:
    #     for line in sequences:
    #         f.write(f"{line}\n")


if __name__ == "__main__":
    
    create_mutated_sequences()