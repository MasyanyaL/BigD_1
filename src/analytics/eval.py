from ultralytics import YOLO


def main():
    model = YOLO("models/best2.pt")

    # Прогоняем валидацию при разных порогах
    for conf in [0.30, 0.50, 0.70]:
        print(f"\n--- Тестирование с conf={conf} ---")

        metrics = model.val(data="configs/dataset1.yaml", conf=conf)

        p = metrics.box.mp
        r = metrics.box.mr
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        print(f"Precision: {p:.3f}")
        print(f"Recall:    {r:.3f}")
        print(f"F1-score:  {f1:.3f}")


if __name__ == '__main__':
    main()