import os
from pathlib import Path
import datetime

def get_size(path):
    """Возвращает размер файла или папки в байтах."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Пропускаем если это симлинк или недоступен
                if not os.path.islink(fp) and os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total
    return 0

def format_size(size):
    """Форматирует размер в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def print_tree(start_path, ignore_list=None, max_depth=None, current_depth=0):
    """Рекурсивно печатает дерево файлов и папок."""
    if ignore_list is None:
        ignore_list = {'.git', '__pycache__', 'node_modules', 'venv', '.idea', '.vscode', 'build', 'dist'}

    if current_depth == 0:
        print(f"{start_path}/")

    
    # Сортируем: сначала папки, потом файлы, и по алфавиту
    items = sorted(os.listdir(start_path), key=lambda x: (not os.path.isdir(os.path.join(start_path, x)), x.lower()))
    
    for item in items:
        if item in ignore_list:
            continue
            
        full_path = os.path.join(start_path, item)
        prefix = "│   " * current_depth
        if item == items[-1]:
            connector = "└── "
        else:
            connector = "├── "
            
        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{item}/")
            if max_depth is None or current_depth < max_depth:
                print_tree(full_path, ignore_list, max_depth, current_depth + 1)
        else:
            size = get_size(full_path)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M')
            print(f"{prefix}{connector}{item} ({format_size(size)}) - {mtime}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    print("Структура проекта:")
    print_tree(base_dir, max_depth=3)  # Ограничим глубину 3 для читаемости