from datasets import load_dataset


def prepare_snli_sentences(tokenizer, limit=None, max_len=64):
    dataset = load_dataset("snli", split="train")
    if limit:
        dataset = dataset.select(range(limit))

    print(f"Processing {len(dataset)} SNLI pairs...")

    sentences = []
    for x in dataset:
        if x['premise']:
            sentences.append(x['premise'])
        if x['hypothesis']:
            sentences.append(x['hypothesis'])

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
        truncation=True
    )
    return inputs['input_ids'], inputs['attention_mask']
