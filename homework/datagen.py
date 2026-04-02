def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # raise NotImplementedError()
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    dataset = Dataset("train")
    model = CoTModel()

    questions = [dataset[i][0] for i in range(len(dataset))]
    correct_answers = [dataset[i][1] for i in range(len(dataset))]

    prompts = [model.format_prompt(q) for q in questions]
    generations = model.batched_generate(
        prompts,
        num_return_sequences=oversample,
        temperature=temperature,
    )

    results = []
    for question, correct_answer, samples in zip(questions, correct_answers, generations):
        for sample in samples:
            parsed = model.parse_answer(sample)
            if parsed == parsed and is_answer_valid(parsed, correct_answer):
                results.append([question, correct_answer, sample.strip()])
                break

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
