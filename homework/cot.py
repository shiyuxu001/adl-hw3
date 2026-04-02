from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        # raise NotImplementedError()
        # messages = [
        #     {"role": "system", "content": "Be concise. Put answer in <answer></answer>."},
        #     {"role": "user", "content": "What is the conversion of 20 yards in inch?"},
        #     {"role": "assistant", "content": "1 yard = 3 feet. 1 foot = 12 inch. 20 * 3 * 12 = <answer>720.0</answer>"},
        #     {"role": "user", "content": question},
        # ]
        messages = [
            {"role": "system", "content": "Be concise. Unit conversion. Put the final answer in <answer></answer>."},
            {"role": "user", "content": "What is the conversion of 2 hour to second?"},
            {"role": "assistant", "content": "2 * 60 * 60 = <answer>7200.0</answer>"},
            {"role": "user", "content": question},
        ]
        # messages = [
        # {
        #     "role": "system",
        #     "content": (
        #         "Convert units."
        #     ),
        # },
        # {
        #     "role": "user",
        #     "content": "How many feet are there per 1 yard?",
        # },
        # {
        #     "role": "assistant",
        #     "content": (
        #         "1 yard = 3 feet. "
        #         "<answer>3</answer>"
        #     ),
        # },
        #     {
        #         "role": "user",
        #         "content": question,
        #     },
        # ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
