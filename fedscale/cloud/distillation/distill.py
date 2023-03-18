"""
Script to distill from n student models into a teacher model.
"""
import argparse
import asyncio
import json
import logging
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.models as tormodels

from fedscale.dataloaders.utils_data import get_data_transform
from fedscale.utils.model_test_module import test_pytorch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distiller")


class DatasetWithIndex(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir')
    parser.add_argument('data_dir')
    parser.add_argument('private_dataset')
    parser.add_argument('public_dataset')
    parser.add_argument('timestamp', type=int)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--student-model', type=str, default=None)
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--acc-check-interval', type=int, default=1)
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()

trained_on_info = {}


def parse_trained_on_info(args):
    # Read the number of samples each model has been trained on
    with open(os.path.join(args.models_dir, "trained_on.json"), "r") as trained_on_file:
        for line in trained_on_file.readlines():
            if not line:
                continue

            json_info = json.loads(line.strip())
            if not json_info["trained_on"]:
                continue

            model_name = json_info["name"]
            parts = model_name.split(".")[0].split("_")
            rank = int(parts[1])
            timestamp = int(parts[2])

            if rank not in trained_on_info:
                trained_on_info[rank] = []
            trained_on_info[rank].append((model_name, timestamp, json_info["trained_on"]))

    # Sort the arrays in trained_on_info based on the timestamp
    for key in trained_on_info.keys():
        trained_on_info[key] = sorted(trained_on_info[key], key=lambda x: x[1])


def get_models_for_timestamp(timestamp):
    """
    Get the model information produced at a particular timestamp
    """
    results = []
    for group in trained_on_info.keys():
        cur_ind = 0
        if trained_on_info[group][0][1] > timestamp: # No module produced at this time!
            raise RuntimeError("Group %d produced no model at time %d!" % (group, timestamp))

        while True:
            if trained_on_info[group][cur_ind][1] > timestamp:
                results.append(trained_on_info[group][cur_ind - 1])
                break
            cur_ind += 1

            if cur_ind == len(trained_on_info[group]):
                results.append(trained_on_info[group][cur_ind - 1])
                break

    return results


async def run(args):
    # Initialize settings
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    with open(os.path.join(args.models_dir, "distill_accuracies.csv"), "w") as out_file:
        out_file.write("epoch,accuracy,loss,best_acc,train_time,total_time\n")

    parse_trained_on_info(args)

    models_info = get_models_for_timestamp(args.timestamp)
    total_per_class = [0] * 10
    for model_info in models_info:
        for cls_label, cls_cnt in model_info[2].items():
            total_per_class[int(cls_label)] += cls_cnt

    print("Total items per class: %s" % total_per_class)

    weights = []
    for model_info in models_info:
        weights_this_group = [None] * 10
        for cls_label, cls_cnt in model_info[2].items():
            weights_this_group[int(cls_label)] = cls_cnt / total_per_class[int(cls_label)]
        weights.append(weights_this_group)

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    start_time = time.time()
    time_for_testing = 0  # Keep track of the time we spend on testing - we want to exclude this

    # Load the private testset and public dataset
    train_transform, test_transform = get_data_transform('cifar')
    test_dataset = CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, args.batch_size)

    cifar100_dataset = CIFAR100(args.data_dir, train=True, download=True, transform=train_transform)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the teacher models
    teacher_models = []
    for model_info in models_info:
        model_name = model_info[0]
        model_path = os.path.join(args.models_dir, model_name)
        if not os.path.exists(model_path):
            raise RuntimeError("Could not find student model at location %s!" % model_path)

        model = tormodels.__dict__['shufflenet_v2_x2_0'](num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        teacher_models.append(model)

        # Test accuracy of the teacher model
        test_loss, acc, acc_5, test_results = test_pytorch_model(0, model, test_loader, device=device)
        logger.info("Accuracy of teacher model %s: %f, %f", model_info[0], acc, test_loss)

    # Create the student model
    student_model = tormodels.__dict__['shufflenet_v2_x2_0'](num_classes=10)
    student_model.to(device)

    # Generate the logits
    logits = []
    for teacher_ind, teacher_model in enumerate(teacher_models):
        teacher_logits = []
        for i, (images, _) in enumerate(cifar100_loader):
            images = images.to(device)
            with torch.no_grad():
                out = teacher_model.forward(images).detach()
                out *= weights[teacher_ind]
            teacher_logits += out

        logits.append(teacher_logits)
        print("Inferred %d outputs for teacher model %d" % (len(teacher_logits), teacher_ind))

    # Aggregate the logits
    print("Aggregating logits...")
    aggregated_predictions = []
    for sample_ind in range(len(logits[0])):
        predictions = [logits[n][sample_ind] for n in range(len(teacher_models))]
        aggregated_predictions.append(torch.sum(torch.stack(predictions), dim=0))

    # Reset loader
    cifar100_loader = DataLoader(dataset=DatasetWithIndex(cifar100_loader.dataset), batch_size=args.batch_size, shuffle=True)

    # Distill \o/
    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss(reduce=True)
    best_acc = 0
    for epoch in range(args.epochs):
        for i, (images, _, indices) in enumerate(cifar100_loader):
            images = images.to(device)

            student_model.train()
            teacher_logits = torch.stack([aggregated_predictions[ind].clone() for ind in indices])
            student_logits = student_model.forward(images)
            loss = criterion(teacher_logits, student_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute the accuracy of the student model
        if epoch % args.acc_check_interval == 0:
            test_start_time = time.time()
            loss, acc, acc_5, test_results = test_pytorch_model(0, student_model, test_loader, device=device)
            if acc > best_acc:
                best_acc = acc
            logger.info("Accuracy of student model after %d epochs: %f, %f (best: %f)", epoch + 1, acc, loss, best_acc)
            time_for_testing += (time.time() - test_start_time)
            with open(os.path.join("data", "distill_accuracies.csv"), "a") as out_file:
                out_file.write("%d,%f,%f,%f,%f,%f\n" % (epoch + 1, acc, loss, best_acc, time.time() - start_time - time_for_testing, time.time() - start_time))

logging.basicConfig(level=logging.INFO)
loop = asyncio.get_event_loop()
loop.run_until_complete(run(get_args()))