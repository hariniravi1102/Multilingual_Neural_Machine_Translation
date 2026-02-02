# inference/decode.py
import torch


@torch.no_grad()
def beam_search_decode(
    model,
    src,
    src_mask,
    max_len,
    sos_idx,
    eos_idx,
    beam_size,
    device
):


    model.eval()

    enc_out = model.encoder(src, src_mask)


    beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            # If EOS already generated, keep beam
            if seq[0, -1].item() == eos_idx:
                new_beams.append((seq, score))
                continue

            T = seq.size(1)

            # Causal mask
            tgt_mask = torch.tril(
                torch.ones(T, T, device=device)
            ).bool().unsqueeze(0).unsqueeze(0)

            # Decode using FULL sequence
            logits, _ = model.decoder(
                tgt=seq,
                enc_out=enc_out,
                tgt_mask=tgt_mask,
                src_mask=src_mask
            )

            # Use last token prediction
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_token = topk_ids[:, i].unsqueeze(1)
                next_seq = torch.cat([seq, next_token], dim=1)

                new_beams.append(
                    (next_seq, score + topk_log_probs[0, i].item())
                )

        # best beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    return beams[0][0]
