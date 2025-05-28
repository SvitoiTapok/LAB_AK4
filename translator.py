#!/usr/bin/python3
"""Транслятор brainfuck в машинный код.
"""

import os
import sys
from cProfile import label

from isa import Opcode, Term, to_bytes, to_hex, opcode_to_binary

memory = [0] * 2 ** 8
labels = {}
number_of_byte = 0
def symbols():
    """Полное множество символов языка brainfuck."""
    return {"not", "read", "write", "add", "sub", "mul", "and", "or", "beq", "bne", "jump", "halt"}


def symbol2opcode(symbol):
    """Отображение операторов исходного кода в коды операций."""
    return {
        "not": Opcode.NOT,
        "read": Opcode.READ,
        "write": Opcode.WRITE,
        "add": Opcode.ADD,
        "sub": Opcode.SUB,
        "mul": Opcode.MUL,
        "and": Opcode.AND,
        "or": Opcode.OR,
        "beq": Opcode.BEQ,
        "bne": Opcode.BNE,
        "jump": Opcode.JUMP,
        "halt": Opcode.HALT,
    }.get(symbol)

def parse_label(label):
    assert label in labels.keys(), f"Неизвестный label {label}"
    new_value = labels.get(label)
    return new_value


def text2terms(text, i):
    """Трансляция текста в последовательность операторов языка (токенов).

    Включает в себя:

    - отсеивание всех незначимых символов (считаются комментариями);
    - проверка формальной корректности программы (парность оператора цикла).
    """
    global labels
    global memory
    global number_of_byte

    terms = []
    while i< len(text):
        val = text[i]
        if val in symbols():
            if val == "halt" or val == "not":
                # memory[number_of_byte] = symbol2opcode(val)
                # number_of_byte+=1
                terms.append([val])
                i+=1
            else:
                # memory[number_of_byte] = symbol2opcode(val)
                # number_of_byte+=1
                # new_value = parse_label(text[i+1])
                # memory[number_of_byte:number_of_byte+4] = int_to_bytes(new_value)
                # number_of_byte+=4
                terms.append([val, text[i+1]])
                i+=2
        if val == ".org":
            terms.append([val, text[i + 1]])
            # number_of_byte += int(text[i + 1])
            i += 2
            continue
        if val.endswith(":"):
            # labels[val[:-1]] = number_of_byte
            terms.append([val])
            i += 1
    return terms



def int_to_bytes(integ):
    return hex(integ%2**8), hex(integ//2**8%2**8), hex(integ//2**16%2**8), hex(integ//2**24%2**8),




def translate(text):
    global labels
    global memory
    global number_of_byte
    """Трансляция текста программы в машинный код.

    Выполняется в два этапа:

    1. Трансляция текста в последовательность операторов языка (токенов).

    2. Генерация машинного кода.

        - Прямое отображение части операторов в машинный код.

        - Отображение операторов цикла в инструкции перехода с учётом
    вложенности и адресации инструкций. Подробнее см. в документации к
    `isa.Opcode`.

    """
    text = text.split()
    i=0
    #макроопределения

    while "#define" in text:
        ind = text.index("#define")
        define_target = text[ind+1]
        define_new_value = text[ind+2]
        text = ' '.join(text).replace(define_target, define_new_value).split()


    #потом подправить
    data_ind = text.index(".data")
    text_ind = text.index(".text")
    assert text_ind!=-1, "Не найдена область .text"
    #обработка переменных
    print("data:", data_ind)
    print("text:", text_ind)
    print(text)



    ind = data_ind+1
    while ind<text_ind:
        if text[ind] == ".org":
            number_of_byte += int(text[ind+1])
            ind += 2
            continue
        lab = text[ind][:-1]
        type = text[ind+1]
        labels[lab] = number_of_byte
        if type == ".num":
            value = int(text[ind+2])
            assert value>2**31-1 or value<2**31, f"Не влезающее в машинное слово число: {value}"
            memory[number_of_byte:number_of_byte+4] = int_to_bytes(value)
            number_of_byte += 4
        if type == ".byte":
            value = text[ind+2]
            for char in value:
                memory[number_of_byte] = hex(ord(char))
                number_of_byte += 1
        ind += 3

    print(labels)
    i = text_ind+1
    terms = text2terms(text, i)

    # Транслируем термы в машинный код.
    code = []
    for pc, term in enumerate(terms):
        val = term[0]
        if val in symbols():
            if val == "halt" or val == "not":
                memory[number_of_byte] =  opcode_to_binary[symbol2opcode(val)]
                code.append({"index": pc, "opcode": symbol2opcode(val), "arg": None})
                number_of_byte+=1
            else:
                memory[number_of_byte] = opcode_to_binary[symbol2opcode(val)]
                number_of_byte+=1
                new_value = parse_label(term[1])
                memory[number_of_byte:number_of_byte+4] = int_to_bytes(new_value)
                number_of_byte+=4
                code.append({"index": pc, "opcode": symbol2opcode(val), "arg": term[1]})
                # begin = {"index": pc, "opcode": Opcode.JZ, "arg": pc + 1, "term": terms[begin_pc]}
                # end = {"index": pc, "opcode": Opcode.JMP, "arg": begin_pc, "term": term}
                # code[begin_pc] = begin
                # code.append(end)
        if val == ".org":
            number_of_byte += int(text[i + 1])
            continue
        if val.endswith(":"):
            labels[val[:-1]] = number_of_byte
    code.append({"index": len(code), "opcode": Opcode.HALT})
    return code, text_ind

def main():
# def main(source, target):
    """Функция запуска транслятора. Параметры -- исходный и целевой файлы."""
    # with open(source, encoding="utf-8") as f:
    #     source = f.read()
    source = ""
    with open("asm_code", encoding="utf-8") as f:
        source = f.read()
    code, text_ind = translate(source)
    # binary_code = to_bytes(code)
    # hex_code = to_hex(code)
    print(code)
    print(memory[:200])
    print(to_hex(memory, text_ind+1, labels))

    # Убедимся, что каталог назначения существует
    # os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)

    # Запишим выходные файлы
    # with open(target, "wb") as f:
    #     f.write(binary_code)
    # with open(target + ".hex", "w") as f:
    #     f.write(hex_code)

    # Обратите внимание, что память данных не экспортируется в файл, так как
    # в случае brainfuck она может быть инициализирована только 0.
    print("source LoC:", len(source.split("\n")), "code instr:", len(code))


if __name__ == "__main__":
    # assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    # _, source, target = sys.argv
    # main(source, target)
    main()