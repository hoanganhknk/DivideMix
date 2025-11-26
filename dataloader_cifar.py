from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle
    with open(file, "rb") as fo:
        d = cPickle.load(fo, encoding="latin1")
    return d


# ===== Noise transition matrices (theo code CIFAR10/100 bạn gửi) =====

def uniform_mix_C(mixing_ratio: float, num_classes: int) -> np.ndarray:
    """
    Linear interpolation của uniform matrix và identity matrix.
    Khi mixing_ratio == r, điều này tương đương 'sym' trong DivideMix gốc.
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1.0 / num_classes) + \
        (1.0 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob: float, num_classes: int, seed: int = 1) -> np.ndarray:
    """
    Mỗi class flip sang đúng 1 class khác với xác suất corruption_prob.
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1.0 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob: float, num_classes: int, seed: int = 1) -> np.ndarray:
    """
    Mỗi class flip sang 2 class khác, chia đều xác suất.
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1.0 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2.0
    return C


class cifar_dataset(Dataset):
    """
    CIFAR dataset dùng cho DivideMix.

    Interface giữ nguyên:
      - mode: 'all', 'labeled', 'unlabeled', 'test'
      - __getitem__:
          'labeled'   -> img1, img2, target, prob
          'unlabeled' -> img1, img2
          'all'       -> img, target, index
          'test'      -> img, target

    Noise mode hỗ trợ:
      - 'sym' hoặc 'unif' : symmetric uniform noise (uniform_mix_C)
      - 'asym'            : asymmetric cố định như DivideMix gốc (CIFAR-10)
      - 'flip'            : random flip 1 class (flip_labels_C)
      - 'flip2'           : random flip 2 class (flip_labels_C_two)
      - 'hierarchical'    : noise theo coarse/fine cho CIFAR-100 (giống code bạn)
    """

    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        root_dir,
        transform,
        mode,
        noise_file="",
        pred=None,
        probability=None,
        log=None,
    ):
        super().__init__()

        if pred is None:
            pred = []
        if probability is None:
            probability = []

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.log = log

        # Bản đồ asymmetric gốc của DivideMix (CIFAR-10)
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

        if self.mode == "test":
            # ===== Load clean test set =====
            if dataset == "cifar10":
                test_dic = unpickle("%s/test_batch" % root_dir)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["labels"]
            elif dataset == "cifar100":
                test_dic = unpickle("%s/test" % root_dir)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["fine_labels"]
        else:
            # ===== Load train set =====
            train_data = []
            train_label = []
            train_coarse_label = None

            if dataset == "cifar10":
                for n in range(1, 6):
                    dpath = "%s/data_batch_%d" % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic["data"])
                    train_label = train_label + data_dic["labels"]
                train_data = np.concatenate(train_data)
                num_classes = 10
            elif dataset == "cifar100":
                train_dic = unpickle("%s/train" % root_dir)
                train_data = train_dic["data"]
                train_label = train_dic["fine_labels"]
                # cho hierarchical corruption
                if "coarse_labels" in train_dic:
                    train_coarse_label = train_dic["coarse_labels"]
                num_classes = 100
            else:
                raise ValueError("Unknown dataset: %s" % dataset)

            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            # ===== Build / load noisy labels =====
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:
                # Tạo noise theo noise_mode
                C = None
                nm = noise_mode
                # 'sym' cũ map sang 'unif'
                if nm == "sym":
                    nm = "unif"

                if nm in ["unif"]:
                    C = uniform_mix_C(self.r, num_classes)
                elif nm == "flip":
                    C = flip_labels_C(self.r, num_classes)
                elif nm == "flip2":
                    C = flip_labels_C_two(self.r, num_classes)
                elif nm == "hierarchical":
                    assert dataset == "cifar100", "hierarchical noise chỉ dùng cho CIFAR-100."
                    assert train_coarse_label is not None, "coarse_labels không tồn tại trong CIFAR-100 train."

                    # Giống hệt đoạn hierarchical trong code CIFAR100 bạn gửi
                    coarse_fine = []
                    for i in range(20):
                        coarse_fine.append(set())
                    for i in range(len(train_label)):
                        coarse_fine[train_coarse_label[i]].add(train_label[i])
                    for i in range(20):
                        coarse_fine[i] = list(coarse_fine[i])

                    C = np.eye(num_classes) * (1.0 - self.r)
                    for i in range(20):
                        tmp = np.copy(coarse_fine[i])
                        for j in range(len(tmp)):
                            tmp2 = np.delete(np.copy(tmp), j)
                            C[tmp[j], tmp2] += self.r * 1.0 / len(tmp2)
                elif nm == "asym":
                    # Giữ nguyên asymmetric cũ cho CIFAR-10
                    noise_label = []
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.r * 50000)
                    noise_idx = set(idx[:num_noise])
                    for i in range(50000):
                        if i in noise_idx:
                            if dataset == "cifar10":
                                noiselabel = self.transition[train_label[i]]
                            else:
                                # CIFAR-100 + asym: để nguyên (hoặc bạn có thể custom thêm sau)
                                noiselabel = train_label[i]
                            noise_label.append(noiselabel)
                        else:
                            noise_label.append(train_label[i])
                    C = None
                else:
                    raise ValueError(
                        "Invalid noise_mode '%s'. Must be in "
                        "{'sym', 'unif', 'asym', 'flip', 'flip2', 'hierarchical'}" % noise_mode
                    )

                # Nếu dùng ma trận C, sample noise giống code bạn
                if C is not None:
                    self.C = C
                    print("Corruption matrix C:\n", C)
                    np.random.seed(1)
                    noise_label = []
                    for y in train_label:
                        new_y = np.random.choice(num_classes, p=C[y])
                        noise_label.append(int(new_y))

                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

            # ===== Split theo mode: all / labeled / unlabeled =====
            if self.mode == "all":
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    # Log AUC giữa prob và true clean/noise
                    if self.log is not None and len(probability) == len(train_label):
                        clean = (np.array(noise_label) == np.array(train_label))
                        auc_meter = AUCMeter()
                        auc_meter.reset()
                        auc_meter.add(np.array(probability), clean.astype(float))
                        auc, _, _ = auc_meter.value()
                        self.log.write("Numer of labeled samples:%d   AUC:%.3f\n" % (pred.sum(), auc))
                        self.log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                else:
                    raise ValueError("Unknown mode %s" % self.mode)

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == "labeled":
            img, target, prob = (
                self.train_data[index],
                self.noise_label[index],
                self.probability[index],
            )
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob

        elif self.mode == "unlabeled":
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2

        elif self.mode == "all":
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index

        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

        else:
            raise ValueError("Unknown mode %s" % self.mode)

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader:
    """
    Wrapper tạo DataLoader cho DivideMix.

    Interface giữ nguyên:
      loader = cifar_dataloader(dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file)
      loader.run('warmup' / 'train' / 'test' / 'eval_train')
    """

    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        batch_size,
        num_workers,
        root_dir,
        log,
        noise_file="",
    ):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        if self.dataset == "cifar10":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ]
            )
        elif self.dataset == "cifar100":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.507, 0.487, 0.441),
                        (0.267, 0.256, 0.276),
                    ),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.507, 0.487, 0.441),
                        (0.267, 0.256, 0.276),
                    ),
                ]
            )
        else:
            raise ValueError("Unknown dataset: %s" % self.dataset)

    def run(self, mode, pred=None, prob=None):
        if pred is None:
            pred = []
        if prob is None:
            prob = []

        if mode == "warmup":
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="all",
                noise_file=self.noise_file,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader

        elif mode == "train":
            labeled_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="labeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                log=self.log,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="unlabeled",
                noise_file=self.noise_file,
                pred=pred,
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            return labeled_trainloader, unlabeled_trainloader

        elif mode == "test":
            test_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="test",
                noise_file=self.noise_file,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="all",
                noise_file=self.noise_file,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

        else:
            raise ValueError("Unknown mode %s" % mode)
