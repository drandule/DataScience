from datetime import datetime
import random

def generate_random_date():
    year = random.randint(2000, 2023)
    month = random.randint(1, 12)

    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(1, 31)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            day = random.randint(1, 29)
        else:
            day = random.randint(1, 28)

    date = f"{year}-{month:02d}-{day:02d}"
    return date

def date_diff(date1, date2):
    try:
        date1 = datetime.strptime(date1, '%Y-%m-%d')
        date2 = datetime.strptime(date2, '%Y-%m-%d')
    except ValueError:
        print("Ошибка формата даты: Используйте формат 'гггг-мм-дд'.")
        return None

    diff = (date2 - date1).days

    if diff < 0:
        diff = abs(diff)

    return diff


# Пример использования функции
if __name__ == "__main__":
    print("Задача 3")
    date_str1 = generate_random_date() #'2023-01-15'
    print(f"Первая дата {date_str1}")
    date_str2 = generate_random_date() #'2023-01-20'
    print(f"Вторая дата {date_str2}")
    diff = date_diff(date_str1, date_str2)
    if diff is not None:
        print(f"Разница в днях : {diff}")