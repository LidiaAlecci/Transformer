import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import helper
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_DIR = os.path.dirname(__file__) + "/content"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = ["numbers__place_value", "comparison__sort", "algebra__linear_1d"]

EXAMPLES_QUESTIONS = [["What is the units digit of 66919584?",
                       "What is the ten thousands digit of 94576916?",
                       "What is the hundreds digit of 54869313?"],
                      ["Sort -1/2, 1/4, 1, 88, 2/889.",
                       "Sort -3, 52, -1, -49, 1, 4, -2 in descending order.",
                       "Put -2/13, 0.1, 4, -1, -1286, -9 in ascending order."],
                      ["Solve 0 = 81*q + 8096 - 4451 for q.",
                       "Solve -1004*k - 10471 + 2408 - 15049 = -362*k for k.",
                       "Solve 27719 = 322*y + 22245 for y."]]
EXAMPLES_ANSWERS = [["4", "7", "3"],
                    ["-1/2, 2/889, 1/4, 1, 88", "52, 4, 1, -1, -2, -3, -49", "-1286, -9, -1, -2/13, 0.1, 4"],
                    ["-45", "-36", "17"]]


class Transformer(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, d_model=256, pad_id=0, sos_id=3, eos_id=2,
                 encoder_layers=3, decoder_layers=2, dim_feedforward=1024, nhead=8):
        super(Transformer, self).__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.embedding_src = nn.Embedding(source_vocabulary_size, d_model, padding_idx=pad_id)
        self.embedding_tgt = nn.Embedding(target_vocabulary_size, d_model, padding_idx=pad_id)

        self.pos_encoder = helper.PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, encoder_layers, decoder_layers, dim_feedforward)
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self, src):
        # Transpose since nn.Transformer takes:
        # src: (S, N, E)
        # src_key_padding_mask: (N, S)
        src_padding_mask = src.transpose(0, 1) == self.pad_id
        return src_padding_mask

    def create_tgt_padding_mask(self, tgt):
        # Transpose since nn.Transformer takes:
        # tgt: (T, N, E)
        # tgt_key_padding_mask: (N, T)
        tgt_padding_mask = tgt.transpose(0, 1) == self.pad_id
        return tgt_padding_mask

    def evaluation(self, source_val, tgt_max):
        output = torch.tensor([], device=DEVICE)
        predictions = torch.tensor([], device=DEVICE)
        for j in range(source_val.size(1)):
            src = source_val[:, j]
            src = src[:, None]
            prediction_prob, prediction = self.greedy_prediction(src, tgt_max)
            output = torch.cat([output, prediction_prob]).to(DEVICE)
            predictions = torch.cat([predictions, prediction[1:]]).to(DEVICE)
        predictions = predictions.reshape(-1, tgt_max-1)
        return predictions, output

    def greedy_prediction(self, src, tgt_max):
        prediction_prob = torch.tensor([], device=DEVICE)
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        src_out = self.embedding_src(src)
        src_out = self.pos_encoder(src_out)
        memory = self.transformer.encoder(src_out, src_key_padding_mask=src_key_padding_mask)
        memory_key_padding_mask = src_key_padding_mask

        prediction = torch.tensor([self.sos_id], device=DEVICE)

        for i in range(tgt_max - 1):
            tgt_tensor = prediction.clone().detach()
            tgt_tensor = tgt_tensor.view(-1)[:, None]
            tgt_key_padding_mask = self.create_tgt_padding_mask(tgt_tensor).to(DEVICE)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(0)).to(DEVICE)

            tgt = self.embedding_tgt(tgt_tensor)
            tgt = self.pos_encoder(tgt)

            output = self.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)
            output = self.linear(output)
            output = output.reshape(-1, output.shape[2])
            # -1 so I will take just the last prediction probability
            prediction_prob = torch.cat([prediction_prob, output[-1]]).to(DEVICE)
            greedy_choice = output[-1].argmax()  # -1 so I will take just the greedy choice of the next character
            prediction = torch.cat([prediction, greedy_choice.view(-1)]).to(DEVICE)

            if greedy_choice == self.eos_id:
                if i < tgt_max - 1 - 1:
                    prediction_pad = torch.zeros(tgt_max).fill_(self.pad_id)
                    prediction_pad[:len(prediction)] = prediction
                    prediction = prediction_pad.to(DEVICE)
                    for j in range(i + 1, tgt_max - 1):
                        prediction_prob = torch.cat([prediction_prob, torch.zeros(output[-1].size(0)).to(DEVICE)]).to(
                            DEVICE)
                break

        return prediction_prob, prediction

    def greedy_prediction_sentence(self, sentence, src_vocab, tgt_vocab, tgt_max, src_max):
        sentence_tensor = torch.tensor([src_vocab.get_idx(char) for char in sentence], device=DEVICE)
        # padding
        src = sentence_tensor.data.new(src_max).fill_(self.pad_id)
        src[:len(sentence_tensor)] = sentence_tensor
        src = src[:, None]

        _, prediction = self.greedy_prediction(src, tgt_max)

        translated_prediction = [tgt_vocab.id_to_string[idx.item()] for idx in prediction]
        return translated_prediction

    def forward(self, src, tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        out = self.transformer(out, tgt, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        out = self.linear(out)
        return out


# Trainer
class Trainer:
    def __init__(self, model, loss_fn, optimizer, task):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.task = task

    def validation(self, valid_data_loader, seq_len_target, tgt_vocab):
        with torch.no_grad():
            loss_val_total = 0
            val_total = 0
            correct_val = 0
            for i, (source_val, target_val) in enumerate(valid_data_loader):
                source_val = source_val.transpose(0, 1).to(DEVICE)
                target_val = target_val.transpose(0, 1).to(DEVICE)
                predictions, output = self.model.evaluation(source_val, tgt_max=target_val.size(0))

                target_val_acc = target_val[1:, :].T
                target_val = target_val_acc.reshape(-1)
                output = output.reshape(-1, len(tgt_vocab))

                loss_val_total += self.loss_fn(output, target_val).item()
                correct_val += torch.sum(torch.sum((target_val_acc == predictions) |
                                                   (target_val_acc == tgt_vocab.pad_id), 1) == seq_len_target - 1)
                val_total += target_val.size(0) // (seq_len_target - 1)

            validation_loss = loss_val_total / len(valid_data_loader)
            validation_accuracy = correct_val * 100 / val_total
        return validation_loss, validation_accuracy

    #  Effective batch size of about 640: accumulate gradients for 10 steps
    def train(self, num_epochs, train_data_loader, valid_data_loader, tgt_vocab, src_vocab, n_step=200,
              stop_accuracy=90):
        prediction_question = ""
        if self.task == TASK[0]:
            prediction_question = "What is the ten thousands digit of 84275766?"
        elif self.task == TASK[1]:
            prediction_question = "Put -25, 45, 13, 2 in increasing order."
        elif self.task == TASK[2]:
            prediction_question = "Solve -12*s + 45 + 3 = -34*s + 7*s + 18 for s."
        self.model.train()
        seq_len_source = next(iter(train_data_loader))[0].size(1)

        steps = []
        training_losses = []
        validation_losses = []
        training_accuracy = []
        validation_accuracy = []
        loss_total = 0
        correct_train = 0
        train_total = 0

        start_time = time.time()
        step = 0
        seq_len_target = next(iter(train_data_loader))[1].size(1)
        for epoch in range(num_epochs):
            for i, (source, target) in tqdm(enumerate(train_data_loader)):
                self.model.train()

                source = source.transpose(0, 1).to(DEVICE)
                target = target.transpose(0, 1).to(DEVICE)

                output = self.model(source, target[:-1, :])

                # I keep the format that I will need for compute the accuracy
                _, output_acc = torch.max(output, dim=2)
                output_acc = output_acc.T
                target_acc = target[1:, :].T
                output = output.reshape(-1, output.shape[2])
                target = target[1:, :].reshape(-1)
                loss = self.loss_fn(output, target)

                loss_total += loss.item()

                loss.backward()  # Compute gradients

                #  Effective batch size of about 640: accumulate gradients for 10 steps
                if i % 10 == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clipping threshold == 1.0
                    self.optimizer.step()  # Update parameters
                    self.optimizer.zero_grad()  # Reset gradients

                correct_train += torch.sum(torch.sum((target_acc == output_acc) | (target_acc == tgt_vocab.pad_id), 1)
                                           == seq_len_target - 1)
                train_total += target.size(0) // (seq_len_target - 1)

                if step and step % n_step == 0:  # Print results every n_step steps
                    training_losses.append(loss_total / len(train_data_loader))
                    training_accuracy.append(correct_train * 100 / train_total)
                    steps.append(step)
                    print(f'step: {step}, train loss: {training_losses[-1]:4.6f}, '
                          f'train accuracy: {training_accuracy[-1]:4.2f} %')
                    with torch.no_grad():
                        prediction_output = self.model.greedy_prediction_sentence(prediction_question, src_vocab,
                                                                                  tgt_vocab, seq_len_target,
                                                                                  seq_len_source)
                    print(f'step: {step}, {prediction_question}: {prediction_output}.')
                    loss_total = 0
                    correct_train = 0
                    train_total = 0

                    self.model.eval()
                    validation_loss, validation_accuracy_val = self.validation(valid_data_loader, seq_len_target,
                                                                               tgt_vocab)
                    validation_losses.append(validation_loss)
                    validation_accuracy.append(validation_accuracy_val)
                    print(f'step: {step}, validation loss: {validation_losses[-1]:4.6f}, validation accuracy: '
                          f'{validation_accuracy[-1]:4.2f} %')
                    if validation_accuracy[-1] > stop_accuracy:
                        minutes_elapsed = round((time.time() - start_time) / 60, 4)
                        print(f'Reach {validation_accuracy[-1]:4.2f} % of validation accuracy after {minutes_elapsed} '
                              f'minutes.')
                        break
                step += 1

            if validation_accuracy[-1] > stop_accuracy:
                print_examples_predictions(self.task, self.model, src_vocab, tgt_vocab, seq_len_target, seq_len_source)
                break

        plot(steps, training_losses, validation_losses, training_accuracy, validation_accuracy)
        return self.model


def get_train_set(task=TASK[0]):
    train_file_name = "train"
    src_file_path = f"{DATASET_DIR}/{task}/{train_file_name}{INPUTS_FILE_ENDING}"
    tgt_file_path = f"{DATASET_DIR}/{task}/{train_file_name}{TARGETS_FILE_ENDING}"
    train_set = helper.ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)
    return train_set


def get_valid_set(task, src_vocab, tgt_vocab):
    valid_file_name = "interpolate"
    src_file_path = f"{DATASET_DIR}/{task}/{valid_file_name}{INPUTS_FILE_ENDING}"
    tgt_file_path = f"{DATASET_DIR}/{task}/{valid_file_name}{TARGETS_FILE_ENDING}"
    valid_set = helper.ParallelTextDataset(src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                           extend_vocab=False)
    return valid_set


def print_examples_predictions(task, model, src_vocab, tgt_vocab, seq_len_target, seq_len_source):
    task_id = -1
    if task == TASK[0]:
        task_id = 0
    elif task == TASK[1]:
        task_id = 1
    elif task == TASK[2]:
        task_id = 2
    for i in range(len(EXAMPLES_QUESTIONS[task_id])):
        question = EXAMPLES_QUESTIONS[task_id][i]
        answer = EXAMPLES_ANSWERS[task_id][i]
        prediction_output = model.greedy_prediction_sentence(question, src_vocab, tgt_vocab, seq_len_target,
                                                             seq_len_source)
        print(f'{question}: \nPrediction:{prediction_output}.\nCorrect answer:{answer}')


def text_data_properties():
    task = TASK[0]
    train_file_name = "train"
    valid_file_name = "interpolate"
    paths = [f"{DATASET_DIR}/{task}/{valid_file_name}{INPUTS_FILE_ENDING}",
             f"{DATASET_DIR}/{task}/{train_file_name}{INPUTS_FILE_ENDING}"]
    sentences_total = 0
    characters_total = 0
    for path in paths:
        with open(path, 'r') as text:
            for sentence_count, line in enumerate(text):
                tokens = list(line)[:-1]  # remove line break
                characters_total += len(tokens)
            sentence_count += 1  # count also the first sentence
        sentences_total += sentence_count
    total_character_questions = characters_total

    paths = [f"{DATASET_DIR}/{task}/{valid_file_name}{TARGETS_FILE_ENDING}",
             f"{DATASET_DIR}/{task}/{train_file_name}{TARGETS_FILE_ENDING}"]
    total_character_answers = 0
    for path in paths:
        with open(path, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                total_character_answers += len(tokens)

    print(f"Number of sentences (train + validation): {sentences_total}.")
    print(f"Number of characters (train + validation): {characters_total}.")
    print(f'Average questions length: {total_character_questions / sentences_total:4.2f}.')
    print(f"Average questions answers: {total_character_answers / sentences_total:4.2f}.")


def run_model(d_model=256, dim_feedforward=1024, task=TASK[0], stop_accuracy=90, n=200):
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-4
    pad_id = 0

    train_set = get_train_set(task)
    valid_set = get_valid_set(task, src_vocab=train_set.src_vocab, tgt_vocab=train_set.tgt_vocab)
    source_vocabulary_size = len(train_set.src_vocab)
    target_vocabulary_size = len(train_set.tgt_vocab)
    train_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)

    model = Transformer(source_vocabulary_size, target_vocabulary_size, d_model=d_model,
                        dim_feedforward=dim_feedforward)

    # Create loss and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore pad_id
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, loss_fn, optimizer, task)
    trainer.train(num_epochs, train_data_loader, valid_data_loader, train_set.tgt_vocab, train_set.src_vocab,
                  stop_accuracy=stop_accuracy, n_step=n)


def plot(steps, train_losses, val_losses, train_accuracy, val_accuracy):
    fig1, ax1 = plt.subplots()
    ax1.plot(steps, train_losses, label="Train loss")
    ax1.plot(steps, val_losses, label="Validation loss")
    ax1.set_xlabel("Step", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(steps, train_accuracy, label="Train accuracy")
    ax2.plot(steps, val_accuracy, label="Validation accuracy")
    ax2.set_xlabel("Step", fontsize=16)
    ax2.set_ylabel("Accuracy", fontsize=16)
    ax2.legend()

    plt.show()


def main():

    text_data_properties()
    run_model()

    run_model(task=TASK[1], n=1000)
    run_model(task=TASK[1], n=1000, stop_accuracy=95)
    run_model(task=TASK[2], n=1000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
