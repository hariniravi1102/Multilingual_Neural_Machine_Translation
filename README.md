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
tests = [
    "I never thought it would end like this.",
    "We don’t have much time—let’s go, now.Are you sure this is the right place?",
    "I did everything I could to save her. No matter what happens, we stay together."
]


Example 1

EN: I never thought it would end like this.
TA: நன்மை தவறாத வழியம் என்று நன்கு கற்றவர்க்கு உண்டு.
HI: मैं इसे सुधार नहीं सकता। मेरा काम हिंदी भाषा में है, और यह एक तमिल वाक्य है।

Example 2

EN: We don’t have much time let’s go, now. Are you sure this is the right place?
TA: குறைந்த காலம் உள்ளவரின் பாசாங்கு இந்த சரியான இடமாகும்
HI: हमें बहुत समय नहीं है, अब आप सुनिश्चित कर लेंगे कि यह सही स्थान है

Example 3

EN: I did everything I could to save her. No matter what happens, we stay together.
TA: அவளகப்பற்றியனாத்தயம்சய்தவட்டன்.
HI: अवलोकन अंततः अनुभवस्यायितर कारण.


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


