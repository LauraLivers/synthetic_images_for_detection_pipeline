from transformers import SegformerForSemanticSegmentation
m = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
print(m.config.id2label)