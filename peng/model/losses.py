
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch
from torch.nn import _reduction as _Reduction
class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        return loss


class Seq2SetLoss(LossBase):
    def __init__(self):
        super().__init__()
        self.Seq2SeqLoss = Seq2SeqLoss()
        self.sample_num = 2

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """
        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        sample_num = self.sample_num
        if pred.shape[0] % sample_num != 0:
            return self.Seq2SeqLoss.get_loss(tgt_tokens, tgt_seq_len, pred)
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        losses = torch.zeros(tgt_tokens.shape[0])
        for i in range(tgt_tokens.shape[0]):
            losses[i] = -F.cross_entropy(target=tgt_tokens[i: i+1], input=pred.transpose(1, 2)[i: i+1])
        losses = losses.reshape(-1, sample_num)
        losses = torch.mean(torch.exp(losses), dim=1)
        log_set_probs = torch.log(losses)
        return -log_set_probs.mean()
