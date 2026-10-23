import os

packages = [
    "src",
    "src/capture",
    "src/inference",
    "src/tracking",
    "src/zones",
    "src/analytics",
    "src/events",
    "src/api",
    "src/logging",
    "src/monitoring"
]

for package in packages:
    init_file_path = os.path.join(package, "__init__.py")
    # Создаём папку, если её нет, и сам файл
    os.makedirs(package, exist_ok=True)
    # Открываем файл в режиме 'a' (добавление), чтобы не стирать его, если он существует
    with open(init_file_path, 'a'):
        pass
    print(f"Создан/проверен: {init_file_path}")