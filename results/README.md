# Артефакты прогона

Файлы этого каталога позволяют перепроверить все числа из отчёта, не запуская модель.

- `preds/` — предсказания на валидации в официальном формате для каждой системы.
  Любой файл можно пере-оценить официальным скриптом:

      PYTHONPATH=src python3 -m ruopin.score \
        --gold RuOpinionNE-2024/validation_labeled.jsonl \
        --pred results/preds/sc480_pred.jsonl

  даёт 0.4603.
- `scores.txt` — F1 по каждому файлу предсказаний (получено тем же скриптом).
- `generations/sc_generations.jsonl` — сырые пять сэмплов на предложение для
  self-consistency. Из них воспроизводится агрегация и свип порога:

      PYTHONPATH=src python3 -m ruopin.aggregate \
        --gen results/generations/sc_generations.jsonl \
        --out /tmp/p.jsonl \
        --gold RuOpinionNE-2024/validation_labeled.jsonl --sweep

- `train_state.json` — история обучения (loss от 0.354 до ~0.03 за три эпохи).
- `logs/` — логи прогонов оценки.

Адаптер LoRA (около 1 ГБ) в репозиторий не залит; он восстанавливается обучением
по `scripts/run.sh`. Файлов предсказаний и логов достаточно, чтобы проверить
итоговые метрики.
