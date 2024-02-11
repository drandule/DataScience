

def word_form(num):
    if num % 10 == 1 and num % 100 != 11:
        return "слово"
    elif num % 10 in [2, 3, 4] and num % 100 not in [12, 13, 14]:
        return "слова"
    else:
        return "слов"


def word_distribution(queries):

    count_dict = {}

    for query in queries:
        count = len(query.split())
        if count in count_dict:
            count_dict[count] += 1
        else:
            count_dict[count] = 1

    total = len(queries)

    # Вывод результатов
    print("Распределение поисковых запросов по количеству слов в каждом из запросов в процентах:")
    for word_count, count in sorted(count_dict.items()):
        percent = (count / total) * 100
        right_word=word_form(word_count)
        print(f"{word_count} {right_word} ({count}): {percent:.2f}%")


# Пример использования функции
if __name__ == "__main__":
    search_queries = ["watch new movies", "coffee near me", "how to find the determinant", "python",
                      "data science jobs in UK", "courses for data science", "taxi", "google", "yandex", "bing",
                      "foreign exchange rates USD/BYN", "Netflix movies watch online free",
                      "Statistics courses online from top universities"]
    print("Задача 5")
    word_distribution(search_queries)