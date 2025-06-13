"""Представление исходного и машинного кода.
"""

from collections import namedtuple
from enum import Enum
import shlex

class Opcode(str, Enum):
    """Opcode для инструкций.

    Можно разделить на две группы:

    1. Непосредственно представленные на уровне языка: `RIGHT`, `LEFT`, `INC`, `DEC`, `INPUT`, `PRINT`.
    2. Инструкции для управления, где:
        - `JMP`, `JZ` -- безусловный и условный переходы:

            | Operator Position | Исходный код | Машинный код |
            |-------------------|--------------|--------------|
            | n                 | `[`          | `JZ (k+1)`   |
            | ...               | ...          |              |
            | k                 |              |              |
            | k+1               | `]`          | `JMP n`      |

        - `HALT` -- остановка машины.
    """
    NOT = "invert bits"
    READ = "read"
    WRITE = "write"
    READ_IND = "read value from address in arg"
    WRITE_IND = "write value from address in arg"
    ADD = "add"
    SUB = "subtract"
    MUL = "multiply"
    AND = "logical and"
    OR = "logical or"
    BEQ = "jump if equal"
    BNE = "jump if negative"
    BVS = "jump if V"
    BCS = "jump if C"
    JUMP = "jump"
    INPUT = "input"
    OUTPUT = "output"
    MUL_HIGH = "high part of mult(high 32 bits)"
    HALT = "halt"

    def __str__(self):
        return str(self.value)


class Term(namedtuple("Term", "line pos symbol")):
    """Описание выражения из исходного текста программы.

    Сделано через класс, чтобы был docstring.
    """


# Словарь соответствия кодов операций их бинарному представлению
opcode_to_binary = {
    Opcode.NOT: '0x00',
    Opcode.READ: '0x01',
    Opcode.WRITE: '0x02',
    Opcode.ADD: '0x03',
    Opcode.SUB: '0x04',
    Opcode.MUL: '0x05',
    Opcode.AND: '0x06',
    Opcode.OR: '0x07',
    Opcode.BEQ: '0x08',
    Opcode.BNE: '0x09',
    Opcode.BVS: '0x0a',
    Opcode.BCS: '0x0b',
    Opcode.JUMP: '0x0c',
    Opcode.INPUT: '0x0d',
    Opcode.OUTPUT: '0x0e',
    Opcode.READ_IND: '0x0f',
    Opcode.WRITE_IND: '0x10',
    Opcode.MUL_HIGH: '0x11',
    Opcode.HALT: '0x12'
}

binary_to_opcode = {
    '0x00':   Opcode.NOT,
    '0x01':  Opcode.READ,
    '0x02': Opcode.WRITE,
    '0x03':   Opcode.ADD,
    '0x04':   Opcode.SUB,
    '0x05':   Opcode.MUL,
    '0x06':  Opcode.AND,
    '0x07':    Opcode.OR,
    '0x08':   Opcode.BEQ,
    '0x09':   Opcode.BNE,
    '0x0a':   Opcode.BVS,
    '0x0b':   Opcode.BCS,
    '0x0c':  Opcode.JUMP,
    '0x0d': Opcode.INPUT,
    '0x0e': Opcode.OUTPUT,
    '0x0f': Opcode.READ_IND,
    '0x10': Opcode.WRITE_IND,
    '0x11': Opcode.MUL_HIGH,
    '0x12': Opcode.HALT
}


def to_bytes(code):
    """Преобразует машинный код в бинарное представление.

    Бинарное представление инструкций:

    ┌─────────┬─────────────────────────────────────────────────────────────┐
    │ 31...24 │ 23                                                        0 │
    ├─────────┼─────────────────────────────────────────────────────────────┤
    │  опкод  │                      адрес перехода                         │
    └─────────┴─────────────────────────────────────────────────────────────┘
    """
    binary_bytes = bytearray()
    for instr in code:
        # Получаем бинарный код операции
        opcode_bin = opcode_to_binary[instr["opcode"]] << 28

        # Добавляем адрес перехода, если он есть
        arg = instr.get("arg", 0)

        # Формируем 32-битное слово: опкод (4 бита) + адрес (28 бит)
        binary_instr = opcode_bin | (arg & 0x00FFFFFF)

        # Преобразуем 32-битное целое число в 4 байта (big-endian)
        binary_bytes.extend(
            ((binary_instr >> 24) & 0xFF, (binary_instr >> 16) & 0xFF, (binary_instr >> 8) & 0xFF, binary_instr & 0xFF)
        )

    return bytes(binary_bytes)


def to_hex(memory, text_ind, labels, data_ind):
    """Преобразует машинный код в текстовый файл с шестнадцатеричным представлением.

    Формат вывода:
    <address> - <HEXCODE> - <mnemonic>
    Например:
    20 - 03340301 - add #01 <- 34 + #03
    """
    revers_labels = {labels[i]: i for i in labels}
    binary_code = memory
    result = []
    i=data_ind
    strin=""
    curlab=''
    while i<text_ind:
        if binary_code[i]==0:
            i+=1
            continue
        if i in revers_labels:
            if strin == "":
                strin = f"{i} - data - "
                curlab = revers_labels[i]
            else:
                strin += f"          {curlab}"
                result.append(strin)
                strin = f"{i} - data - "
                curlab = revers_labels[i]
        byte = binary_code[i][2:]
        if len(byte) == 1:
            byte = '0' + byte
        strin += byte
        i += 1
    strin += f"          {curlab}"
    result.append(strin)

    i=text_ind
    while i<len(binary_code):
        if binary_code[i]==0:
            i+=1
            continue
        strin = f"{i}"
        opcode_instr = binary_code[i]
        i += 1
        opcode_instr_1 = binary_to_opcode.get(opcode_instr)

        if opcode_instr_1==Opcode.HALT or opcode_instr_1==Opcode.NOT:
            strin += f" - {opcode_instr} - {Opcode(opcode_instr_1)}"
            if not revers_labels.get(i - 1) is None:
                strin += f"          {revers_labels[i - 1]}"
            result.append(strin)
            continue
        arg = ""
        for ind in range(4):
            byte = binary_code[i+ind][2:]
            if len(byte)==1:
                byte = '0' + byte
            arg+=byte
        i += 4
        strin += f" - {opcode_instr} {arg} - {Opcode(opcode_instr_1)}"
        if not revers_labels.get(i-5) is None:
            strin += f"          {revers_labels[i-5]}"
        result.append(strin)
    return "\n".join(result)


def from_bytes(binary_code):
    """Преобразует бинарное представление машинного кода в структурированный формат.

    Бинарное представление инструкций:

    ┌─────────┬─────────────────────────────────────────────────────────────┐
    │ 31...28 │ 27                                                        0 │
    ├─────────┼─────────────────────────────────────────────────────────────┤
    │  опкод  │                      адрес перехода                         │
    └─────────┴─────────────────────────────────────────────────────────────┘
    """
    structured_code = []
    # Обрабатываем байты по 4 за раз для получения 32-битных инструкций
    for i in range(0, len(binary_code), 4):
        if i + 3 >= len(binary_code):
            break

        # Формируем 32-битное слово из 4 байтов
        binary_instr = (
            (binary_code[i] << 24) | (binary_code[i + 1] << 16) | (binary_code[i + 2] << 8) | binary_code[i + 3]
        )

        # Извлекаем опкод (старшие 4 бита)
        opcode_bin = (binary_instr >> 28) & 0xF
        opcode = binary_to_opcode[opcode_bin]

        # Извлекаем адрес перехода (младшие 28 бит)
        arg = binary_instr & 0x0FFFFFFF

        # Формируем структуру инструкции
        instr = {"index": i // 4, "opcode": opcode}

        # Добавляем адрес перехода только для инструкций перехода
        if opcode in (Opcode.JMP, Opcode.JZ):
            instr["arg"] = arg

        structured_code.append(instr)

    return structured_code