"""RNN encoder-decoder models for abstractive summarization."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        pad_token_id: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.dropout(self.embedding(input_ids))
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_output_dim: int, decoder_hidden_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_output_dim, decoder_hidden_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.score = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(
            self.encoder_proj(encoder_outputs)
            + self.decoder_proj(decoder_hidden).unsqueeze(1)
        )
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(src_mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
        use_attention: bool,
        encoder_output_dim: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.attention = (
            BahdanauAttention(encoder_output_dim, hidden_dim) if use_attention else None
        )
        lstm_input_dim = embedding_dim + (encoder_output_dim if use_attention else 0)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        if use_attention:
            self.output = nn.Linear(hidden_dim + encoder_output_dim + embedding_dim, vocab_size)
        else:
            self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ):
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
        attention_weights = None

        if self.use_attention:
            context, attention_weights = self.attention(
                hidden[-1],
                encoder_outputs,
                src_mask,
            )
            rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            context = None
            rnn_input = embedded

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = output.squeeze(1)

        if self.use_attention:
            logits = self.output(torch.cat([output, context, embedded.squeeze(1)], dim=-1))
        else:
            logits = self.output(output)
        return logits, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    """Configurable LSTM/BiLSTM encoder-decoder with optional attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False,
        pad_token_id: int = 0,
        start_token_id: int = 0,
        eos_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.eos_token_id = eos_token_id

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_token_id=pad_token_id,
        )
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.hidden_bridge = nn.Linear(encoder_output_dim, hidden_dim)
        self.cell_bridge = nn.Linear(encoder_output_dim, hidden_dim)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_attention=use_attention,
            encoder_output_dim=encoder_output_dim,
        )

    def _merge_directions(self, state: torch.Tensor) -> torch.Tensor:
        if not self.bidirectional:
            return state
        batch_size = state.size(1)
        state = state.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        return torch.cat([state[:, 0], state[:, 1]], dim=-1)

    def _match_decoder_layers(self, state: torch.Tensor) -> torch.Tensor:
        if state.size(0) >= self.decoder_layers:
            return state[-self.decoder_layers :]
        repeat_count = self.decoder_layers - state.size(0)
        return torch.cat([state, state[-1:].repeat(repeat_count, 1, 1)], dim=0)

    def _init_decoder_state(self, hidden: torch.Tensor, cell: torch.Tensor):
        hidden = self._merge_directions(hidden)
        cell = self._merge_directions(cell)
        hidden = torch.tanh(self.hidden_bridge(hidden))
        cell = torch.tanh(self.cell_bridge(cell))
        hidden = self._match_decoder_layers(hidden).contiguous()
        cell = self._match_decoder_layers(cell).contiguous()
        return hidden, cell

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs, hidden, cell = self.encoder(input_ids, attention_mask)
        hidden, cell = self._init_decoder_state(hidden, cell)
        return encoder_outputs, hidden, cell

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, target_len = labels.size()
        outputs = torch.zeros(
            batch_size,
            target_len,
            self.vocab_size,
            device=input_ids.device,
        )
        encoder_outputs, hidden, cell = self.encode(input_ids, attention_mask)
        input_token = torch.full(
            (batch_size,),
            self.start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        for t in range(target_len):
            logits, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs=encoder_outputs,
                src_mask=attention_mask,
            )
            outputs[:, t] = logits
            predicted = logits.argmax(dim=-1)
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = labels[:, t] if use_teacher else predicted
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int = 100,
    ) -> torch.Tensor:
        self.eval()
        batch_size = input_ids.size(0)
        encoder_outputs, hidden, cell = self.encode(input_ids, attention_mask)
        input_token = torch.full(
            (batch_size,),
            self.start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        generated = []

        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs=encoder_outputs,
                src_mask=attention_mask,
            )
            input_token = logits.argmax(dim=-1)
            generated.append(input_token)
            if self.eos_token_id is not None:
                finished |= input_token.eq(self.eos_token_id)
                if finished.all():
                    break

        if not generated:
            return torch.empty(batch_size, 0, dtype=torch.long, device=input_ids.device)
        return torch.stack(generated, dim=1)


def build_model(model_config: dict) -> Seq2Seq:
    return Seq2Seq(**model_config)

