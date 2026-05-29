from .serialize import opinions_to_target

SYSTEM = (
    "Ты извлекаешь мнения из новостного предложения на русском языке.\n"
    "Мнение — это кортеж из четырёх полей:\n"
    "- holder: кто высказывает отношение. Точная подстрока предложения (именованная сущность), "
    "либо \"AUTHOR\", если отношение исходит от автора текста, либо \"NULL\", если источник не указан.\n"
    "- target: к кому или чему отношение. Точная подстрока предложения.\n"
    "- expression: слова предложения, обосновывающие тональность. Список из одной или нескольких точных подстрок.\n"
    "- polarity: \"POS\" для положительного отношения, \"NEG\" для отрицательного.\n"
    "Выводи только JSON-массив таких объектов, без пояснений. Если мнений нет, выводи [].\n"
    "Все holder/target/expression должны дословно встречаться в предложении."
)

FEWSHOT = [
    (
        "Премьер-министр Италии Маттео Ренци заявил, что с уходом Фо «страна потеряла "
        "крупнейшую фигуру культурной жизни»",
        '[{"holder": "Маттео Ренци", "target": "Фо", "expression": ["крупнейшую фигуру культурной жизни"], "polarity": "POS"}]',
    ),
    (
        "Вчера он уволил Азамата Сагитова, который возглавил башкирскую администрацию год назад.",
        '[{"holder": "NULL", "target": "Азамата Сагитова", "expression": ["уволил"], "polarity": "NEG"}]',
    ),
    (
        "В числе участников президентской борьбы есть одна женщина — Айссата Хайдара Сиссе.",
        "[]",
    ),
]


def user_msg(text):
    return f"Предложение: {text}"


def build_messages(text, n_shots=0):
    msgs = [{"role": "system", "content": SYSTEM}]
    for shot_text, shot_out in FEWSHOT[:n_shots]:
        msgs.append({"role": "user", "content": user_msg(shot_text)})
        msgs.append({"role": "assistant", "content": shot_out})
    msgs.append({"role": "user", "content": user_msg(text)})
    return msgs


def build_train_messages(sentence):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg(sentence["text"])},
        {"role": "assistant", "content": opinions_to_target(sentence["opinions"])},
    ]
    return msgs
