'''
Задача 4:
Реализуйте двусвязный список используя синтаксис языка Python. Вам необходимо создать класс (либо несколько классов), который (которые) будет (будут)
представлять структуру данных - связный список.
Связный список — это набор элементов данных, называемых узлами. В односвязном списке каждый узел содержит значение и ссылку на следующий узел.
В двусвязном списке каждый узел также содержит ссылку на предыдущий узел.
Реализуйте узел для хранения значения и указателей на следующий и предыдущий узлы.
Затем реализуйте список, который содержит ссылки на первый и последний узел и предлагает интерфейс, подобный массиву, для добавления и удаления элементов,
какие методы должны быть реализованы:

push() - записывает значение в конец списка
pop() - удаляет значение с конца списка
shift() - удаляет значение в начале списка
unshift() - записывает значение в начало списка

'''

class DoublyLinkedListNode:
    def __init__(self,data):
        self.data = data;
        self.previous = None;
        self.next = None;

class DoublyLinkedList:
    def __init__(self):
        self.head = None;
        self.tail = None

    def push(self, data):
        new_node = DoublyLinkedListNode(data)
        if not self.head:
            self.head = self.tail=new_node
            self.head.previous=None
            self.tail.next=None
        else:
            self.tail.next=new_node
            new_node.previous=self.tail
            self.tail=new_node
            self.tail.next=None

    def pop(self):
        if not self.tail:
            return None
        pop_data=self.tail.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.previous
            self.tail.next = None
        return pop_data


    def shift(self):
        if not self.head:
            return None
        pop_data=self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.previous = None
        return pop_data

    def unshift(self, data):
        new_node = DoublyLinkedListNode(data)
        if not self.head:
            self.head = self.tail=new_node
            self.head.previous=None
            self.tail.next=None
        else:
            self.head.previous=new_node
            new_node.next=self.head
            self.head=new_node
            self.head.previous=None

if __name__ == "__main__":

    print("Задача 4")

    linkedlist = DoublyLinkedList()

    linkedlist.push(1)
    linkedlist.push(2)
    linkedlist.push(3)

    print(linkedlist.pop())
    print(linkedlist.shift())
    linkedlist.unshift(4)
    print(linkedlist.shift())