# please refer the official implementation of Pyabsa - https://github.com/yangheng95/PyABSA

from pyabsa import AspectSentimentTripletExtraction as ASTE

from pyabsa import available_checkpoints

ckpts = available_checkpoints(
    show_ckpts=True
)
triplet_extractor = ASTE.AspectSentimentTripletExtractor(
    checkpoint="english"
)
examples = [
    "maggi is delicious",
    "maggi is too spicy"
]
for example in examples:
    triplet_extractor.predict(example)