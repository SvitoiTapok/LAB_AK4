#!/usr/bin/python3
"""Транслятор brainfuck в машинный код.
"""

import os
import re
import shlex
import sys

import numpy as np

from isa import Opcode, to_hex, opcode_to_binary

memory = [0] * 2 ** 16
labels = {}
number_of_byte = 0
def symbols():
    """Полное множество символов языка brainfuck."""
    return {"not", "read", "write", "add", "sub", "mul", "and", "or", "beq", "bne","bvs", "bcs", "jump", "read_ind", "write_ind", "mul_high", "halt"}


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
        "bvs": Opcode.BVS,
        "bcs": Opcode.BCS,
        "jump": Opcode.JUMP,
        "read_ind": Opcode.READ_IND,
        "write_ind": Opcode.WRITE_IND,
        "mul_high": Opcode.MUL_HIGH,
        "halt": Opcode.HALT,
    }.get(symbol)

def parse_label(label):
    assert label in labels.keys(), f"Неизвестный label {label}"
    new_value = labels.get(label)
    return new_value


def text2terms(text, i):
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
            continue
        if val == ".org":
            terms.append([val, text[i + 1]])
            # number_of_byte += int(text[i + 1])
            i += 2
            continue
        if val.endswith(":"):
            # labels[val[:-1]] = number_of_byte
            terms.append([val])
            i += 1
            continue
        assert 0, f"unexpected character: {val}"
    return terms



def int_to_bytes(integ):
    first = hex(integ%2**8)
    second = hex(integ//2**8%2**8)
    third = hex(integ//2**16%2**8)
    fourth = hex(integ//2**24%2**8)
    if len(first)==3: first = '0x0' + first[-1]
    if len(second)==3: second = '0x0' + second[-1]
    if len(third)==3: third = '0x0' + third[-1]
    if len(fourth)==3: fourth = '0x0' + fourth[-1]
    return first, second, third, fourth
def byte_to_int(bytes):
    return int.from_bytes([int(x, 16) for x in bytes], byteorder='little', signed=True)
# def int_to_chars(integ):
#     first = chr(integ%2**8)
#     second = chr(integ//2**8%2**8)
#     third = chr(integ//2**16%2**8)
#     fourth = chr(integ//2**24%2**8)
#     return first, second, third, fourth
def int_to_ints(integ):
    first = integ%2**8
    second = integ//2**8%2**8
    third = integ//2**16%2**8
    fourth = integ//2**24%2**8
    return first, second, third, fourth



def translate(text):
    global labels
    global memory
    global number_of_byte
    memory = [0] * 2 ** 16
    labels = {}
    number_of_byte = 8

    text = re.sub(r';.*$', '', text, flags=re.MULTILINE)
    text = shlex.split(text, posix=False)
    i=0
    #макроопределения

    while "#def" in text:
        ind = text.index("#def")
        end = text.index("#enddef")
        define_target = text[ind+1]
        define_number_of_values = int(text[ind+2])
        define_value = ' '.join(text[ind+3:end])
        text = text[:ind] + text[end+1:]
        while define_target in text:
            target_ind = text.index(define_target)
            vals = []
            while define_number_of_values:
                vals.append(text[target_ind+1+len(vals)])
                define_number_of_values -= 1
            for val in range(len(vals)):
                define_value = define_value.replace(f'%{val+1}', vals[val])
            define_value = shlex.split(define_value, posix=False)
            text = text[:target_ind] + define_value + text[target_ind+len(vals)+1:]
    data_ind = text.index(".data")
    text_ind = text.index(".text")


    #первые 8 байт зарезервированны под потоки ввода-вывода
    memory_data_ind = 0
    ind = data_ind+1
    fl = True
    while ind<text_ind:
        if text[ind] == ".org":
            number_of_byte += int(text[ind+1])
            ind += 2
            continue
        if fl:
            memory_data_ind = number_of_byte
            fl = False
        lab = text[ind][:-1]
        type = text[ind+1]
        labels[lab] = number_of_byte
        if type == ".num":
            value = int(text[ind+2])
            assert value>2**31-1 or value<2**31, f"Не влезающее в машинное слово число: {value}"
            memory[number_of_byte:number_of_byte+4] = int_to_bytes(value)
            number_of_byte += 4
        if type == ".byte":
            value = text[ind+2][1:-1]
            for char in value:
                memory[number_of_byte] = hex(ord(char))
                number_of_byte += 1
        ind += 3



    i = text_ind+1
    terms = text2terms(text, i)
    memory_text_ind = number_of_byte
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
                number_of_byte+=4
                code.append({"index": pc, "opcode": symbol2opcode(val), "arg": term[1]})
        if val == ".org":
            number_of_byte += int(text[i + 1])
            continue
        if val.endswith(":"):
            labels[val[:-1]] = number_of_byte
    number_of_byte = memory_text_ind


    #для корректного парса лэйблов
    for pc, term in enumerate(terms):
        val = term[0]
        if val in symbols():
            if val == "halt" or val == "not":
                number_of_byte += 1
            else:
                number_of_byte += 1
                new_value = parse_label(term[1])
                memory[number_of_byte:number_of_byte+4] = int_to_bytes(new_value)
                number_of_byte += 4
        if val == ".org":
            number_of_byte += int(text[i + 1])
    assert "_start" in labels, "Не найдена метка _start"
    code.append({"index": len(code), "opcode": Opcode.HALT})
    return code, memory_text_ind, memory_data_ind, labels.get("_start")

def main(source, target):
    """Функция запуска транслятора. Параметры -- исходный и целевой файлы."""
    # with open(source, encoding="utf-8") as f:
    #     source = f.read()
    with open(source, encoding="utf-8") as f:
        source = f.read()
    code, text_ind, data_ind, start_ind = translate(source)
    # binary_code = to_bytes(code)
    # hex_code = to_hex(code)
    # print(code)
    # print(memory[:200])
    # print(start_ind)
    # print(to_hex(memory, text_ind, labels, data_ind))

    # Убедимся, что каталог назначения существует
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)

    # Запишим выходные файлы
    with open(target, "wb") as f:
        numbers = [int(x, 16) for x in int_to_bytes(start_ind)]
        # print(int_to_bytes(start_ind))
        # print(start_ind)
        byte_data = bytes(numbers)
        f.write(byte_data)
        numbers = []
        for x in memory:
            if type(x)==str:
                numbers.append(int(x, 16))
            else:
                numbers.append(x)
        byte_data = bytes(numbers)
        f.write(byte_data)
    with open(target + ".hex", "w") as f:
        f.write(to_hex(memory, text_ind, labels, data_ind))

    # Обратите внимание, что память данных не экспортируется в файл, так как
    # в случае brainfuck она может быть инициализирована только 0.
    print("source LoC:", len(source.split("\n")), "code instr:", len(code))


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Wrong arguments: translator.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)