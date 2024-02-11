import random

def generate_list(length,upto):
    return [random.randint(1, upto) for _ in range(length)]


def vector_dot(vector1, vector2):
    if len(vector1) != len(vector2):
        print("Ошибка: Векторы должны иметь одинаковую длину.")
        return None
    result = sum(a * b for a, b in zip(vector1, vector2))
    return result


def vector_length(vector):
    squared_length = sum(v ** 2 for v in vector)
    length = squared_length ** 0.5
    return length

def cosine_distance(vector1,vector2):
    length1 = vector_length(vector1)
    length2 = vector_length(vector2)

    if length1 == 0 or length2 == 0:
        print("Ошибка: Длины векторов не должны быть равны 0.")
        return None

    dot = vector_dot(vector1, vector2)
    result = dot / (length1 * length2)
    return result


if __name__ == '__main__':
    print("Задача 2")
    vector1=generate_list(10,10)
    vector2=generate_list(10,10)
    #print(vector1)
    #print(vector2)
    distance=cosine_distance(vector1,vector2)
    print(f"косинусная дистанция = {distance}")