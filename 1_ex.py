import random

def generate_list(length,upto):
    return [random.randint(1, upto) for _ in range(length)]

def unique_sorted(array):
    unique_set=set(array)
    unique_sorted_list=sorted(unique_set)
    return unique_sorted_list


if __name__ == '__main__':
    print("Задача 1")
    array=generate_list(10,10)
    print(array)
    sorted_array=unique_sorted(array)
    print(sorted_array)
