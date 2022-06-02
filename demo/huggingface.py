from transformers import pipeline

# generator = pipeline(task='text-generation')
# reuslt = generator(
#     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
# )

from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# model = AutoModelForCausalLM.from_pretrained('distilgpt2')
# generator = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
# result = generator(
#     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
# ) 
# print(result)

# vision_classifier = pipeline(task='image-classification')
# preds = vision_classifier(
#     images='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg'
# )
# preds = [{'score': round(pred['score'], 4), 'label': pred['label']} for pred in preds]
# print(preds)


# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# sequences = 'I love you'
# encoded_input = tokenizer(sequences)
# print(tokenizer.decode(encoded_input['input_ids']))
# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True, return_tensors='pt')
# print(encoded_input)

from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor

# dataset = load_dataset('imagefolder', data_dir='../data/test')
# # print(dataset['train'][0]['image'])
# # plt.imshow(dataset['train'][0]['image'])
# # plt.show()

# feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
# _transforms = Compose([RandomResizedCrop(feature_extractor.size),
#                        ColorJitter(brightness=0.5, hue=0.5), 
#                        ToTensor(), 
#                        normalize])

# def transforms(examples):
#     examples['pixel_values'] = [_transforms(image.convert('RGB')) for image in examples['image']]
#     return examples

# dataset.set_transform(transforms)
# img = dataset['train'][1]['pixel_values']
# print(img.shape)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()


from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

dataset = load_dataset('yelp_review_full')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
training_args = TrainingArguments(output_dir='./output/', evaluation_strategy='epoch')

metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

# trainer.train()
print(tokenized_datasets)


