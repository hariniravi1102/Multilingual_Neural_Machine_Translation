# Multilingual Neural Machine Translation (EN to HI, EN to TA)

A  multilingual Neural Machine Translation (NMT) system built
from scratch using a Transformer encoder–decoder architecture in PyTorch.

Supports:
- English to Hindi
- English to Tamil

The system uses a shared SentencePiece tokenizer, language tag conditioning,
beam search decoding, and optional LLaMA based post-editing.



## Features

- Transformer encoder–decoder 
- Multilingual training (EN to HI + EN to TA)
- Shared SentencePiece unigram tokenizer
- Explicit language tags: `<2hi>`, `<2ta>`
- Beam search decoding
- Optional LLaMA post-edit refinement
- GPU-accelerated training and inference


### Multilingual Neural Machine Translation (EN to HI, EN to TA)

A multilingual Neural Machine Translation (NMT) system built
from scratch using a Transformer encoder–decoder architecture in PyTorch.

Supports:
- English to Hindi
- English to Tamil

The system uses a shared SentencePiece tokenizer, language-tag conditioning,
beam search decoding, and optional LLaMA-based post-editing.



## Features

- Transformer encoder–decoder
- Multilingual training (EN to HI + EN to TA)
- Shared SentencePiece unigram tokenizer
- Explicit language tags: `<2hi>`, `<2ta>`
- Beam search decoding
- LLaMA post-edit refinement
- GPU-accelerated training and inference



### Test Sentences
```python



EN : How are you today?
TA : நீ இன்று எப்படி இருக்கிறாய்?
HI : आज आपका दिन कैसा जा रहा है?

EN : The weather is nice today.
TA : வெயில் சிறப்பானது.
HI : आज का मौसम अच्छा है।

EN : I want to play cricket.
TA : கிரிக்கெட் விளையாடும் பழக்கத்தை எனக்குண்டு.
HI : मैं क्रिकेट खेलने का इच्छुक हूँ।





```
## Datasets

This project is trained entirely on **publicly available open-source datasets**.

### English → Hindi
- **IIT Bombay English–Hindi Corpus**
- Released by IIT Bombay CFILT


### English → Tamil
- **OpenSubtitles (EN–TA)**
- Public movie subtitle corpus

### Notes
- All datasets are used for **research and educational purposes**
- No proprietary or private data is included
- Data was preprocessed, cleaned, and tokenized using SentencePiece


