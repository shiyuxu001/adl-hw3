from homework.cot import CoTModel

m = CoTModel()
qs = [
    "How many gram are there per 6 kg?",
    "How many feet are there per 2 yard?",
    "How many inch are there per 3 foot?",
]

for q in qs:
    prompt = m.format_prompt(q)
    out = m.generate(prompt)
    print("Q:", q)
    print("OUT:", repr(out))
    print("PARSED:", m.parse_answer(out))
    print()