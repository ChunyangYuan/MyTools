class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def __repr__(self):
        repr_str = f"Book(title='{self.title}', author='{self.author}', year={self.year})"
        return repr_str

# 创建两本书的实例
book1 = Book("The Great Gatsby", "F. Scott Fitzgerald", 1925)
book2 = Book("To Kill a Mockingbird", "Harper Lee", 1960)

# 打印书的实例
print(book1)
print(book2)
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 创建一个 Person 实例
person = Person("Alice", 30)

# 打印 Person 实例
print(person)
