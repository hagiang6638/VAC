import torch
import numpy as np
from itertools import groupby
from pyctcdecode import build_ctcdecoder


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

        # Fake vocab giống code cũ (mỗi class = 1 token)
        self.vocab = [str(i) for i in range(num_classes)]

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            blank_token=self.vocab[blank_id],
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        nn_output: (B, T, C)
        """
        if not probs:
            nn_output = torch.softmax(nn_output, dim=-1)

        nn_output = nn_output.cpu().numpy()
        vid_lgt = vid_lgt.cpu().numpy()

        ret_list = []

        for b in range(nn_output.shape[0]):
            logits = nn_output[b][:vid_lgt[b]]

            decoded = self.ctc_decoder.decode(logits)

            # decoded là string "0 3 5 ..."
            ids = [int(x) for x in decoded.split()] if decoded else []

            # remove duplicate (CTC collapse)
            ids = [x for x, _ in groupby(ids)]

            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(ids)
            ])

        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, _ = index_list.shape
        ret_list = []

        for batch_idx in range(batchsize):
            group_result = [
                x[0]
                for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])
            ]

            filtered = list(filter(lambda x: x != self.blank_id, group_result))

            if len(filtered) > 0:
                max_result = [x[0] for x in groupby(filtered)]
            else:
                max_result = []

            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(max_result)
            ])

        return ret_list
