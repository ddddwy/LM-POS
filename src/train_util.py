from torch.autograd import Variable
import numpy as np
import torch
import config

def get_input_from_batch(batch, use_cuda):
  batch_size = len(batch.enc_lens)
  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens
  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()
    c_t_1 = c_t_1.cuda()

  return enc_batch, enc_padding_mask, enc_lens, c_t_1

def get_output_from_batch(batch, use_cuda):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()

  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

