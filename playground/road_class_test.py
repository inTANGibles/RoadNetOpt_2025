class MyClass:
    __value = 0

    @classmethod
    def change_value(cls, value):
        cls.__value = value

    @classmethod
    def show_value(cls):
        print(cls.__value, cls.__name__)


class NewClass(MyClass):
    pass


MyClass.show_value()
NewClass.show_value()

NewClass.change_value(1)

MyClass.show_value()
NewClass.show_value()

del NewClass

NewClass.show_value()
