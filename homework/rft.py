from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    # raise NotImplementedError()
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .data import Dataset
    from .sft import TokenizedDataset

    trainset = Dataset("rft")
    llm = BaseLLM()

    def format_example(prompt: str, answer: float, reasoning: str) -> dict[str, str]:
        return {
            "question": prompt,
            "answer": reasoning,
        }

    lora_config = LoraConfig(
        r=24,
        lora_alpha=96,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    tokenized_trainset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_trainset,
    )

    trainer.train()
    trainer.save_model(Path(__file__).parent / "rft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
