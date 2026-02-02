#dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0


class ParallelTextDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src = self._load_file(src_file)
        self.tgt = self._load_file(tgt_file)
        assert len(self.src) == len(self.tgt), "Source and target size mismatch"

    def _load_file(self, path):
        sequences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ids = list(map(int, line.strip().split()))
                sequences.append(torch.tensor(ids, dtype=torch.long))
        return sequences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(
        src_batch, batch_first=True, padding_value=PAD_ID
    )
    tgt_padded = pad_sequence(
        tgt_batch, batch_first=True, padding_value=PAD_ID
    )

    decoder_input = tgt_padded[:, :-1]
    decoder_target = tgt_padded[:, 1:]

    src_mask = (src_padded != PAD_ID)
    tgt_mask = (decoder_input != PAD_ID)

    return {
        "src": src_padded,
        "tgt_input": decoder_input,
        "tgt_output": decoder_target,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def build_dataloader(
    src_file,
    tgt_file,
    batch_size,
    shuffle=True,
    num_workers=0
):
    dataset = ParallelTextDataset(src_file, tgt_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    return loader
