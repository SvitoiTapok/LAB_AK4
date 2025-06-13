#!/usr/bin/python3
"""Модель процессора, позволяющая выполнить машинный код полученный из программы
на языке Brainfuck.

Модель включает в себя три основных компонента:

- `DataPath` -- работа с памятью данных и вводом-выводом.

- `ControlUnit` -- работа с памятью команд и их интерпретация.

- и набор вспомогательных функций: `simulation`, `main`.
"""

import logging
import sys
from enum import Enum
from unittest.mock import right
import warnings
import numpy as np

from isa import Opcode, from_bytes, opcode_to_binary, binary_to_opcode
from translator import byte_to_int, int_to_bytes, int_to_chars
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Selector(str, Enum):
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
    PC="PС"
    ACC="ACC"
    ALU="ALU"
    DATA_OPERAND="DO"
    ADD1 = "ADD1"
    ADD4 = "ADD4"
    IN1="IN1"
    IN2="IN2"
    LUT="LUT"

    def __str__(self):
        return str(self.value)


class ALU:
    left_input = None
    right_input = None
    out = None
    C = None
    V = None
    def __init__(self):
        self.left_input=np.int32(0)
        self.right_input=np.int32(0)
        self.out=np.int32(0)
        self.C=False
        self.V=False

    def signal_add(self):
        x = self.left_input
        y = self.right_input
        self.out = x + y
        sum_uint = x.astype(np.uint32) + y.astype(np.uint32)
        self.C = sum_uint < x.astype(np.uint32)
        self.V = ((x ^ y) >= 0) & ((x ^ self.out) < 0)

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)

    def signal_sub(self):
        x = self.right_input
        y = -self.left_input
        self.out = x + y
        sum_uint = x.astype(np.uint32) + y.astype(np.uint32)
        self.C = sum_uint < x.astype(np.uint32)
        self.V = ((x ^ y) >= 0) & ((x ^ self.out) < 0)

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)

    def signal_mul(self):
        x = self.right_input
        y = self.left_input
        self.out = x * y
        sum_int = x.astype(int) * y.astype(int)
        self.V = sum_int > self.out

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)
    def signal_mul_high(self):
        x = self.right_input
        y = self.right_input
        mul_int = x.astype(int) * y.astype(int) // 2**32
        self.out = mul_int

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)

    def signal_not(self):
        self.out = ~self.out

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)

    def signal_or(self):
        self.out = self.right_input | self.left_input

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)

    def signal_and(self):
        self.out = self.right_input & self.left_input

        self.left_input = np.int32(0)
        self.right_input = np.int32(0)




class DataPath:

    data_memory_size = None
    "Размер памяти данных."
    addres_register = None
    program_counter = None
    ALU = None

    data = None
    "Память."

    acc = None
    "Аккумулятор. Инициализируется нулём."
    data_operand = None

    input_buffer1 = None
    input_buffer2 = None

    output_buffer1 = None
    output_buffer2 = None
    "Буфер выходных данных."

    def __init__(self, data_memory_size, data, input_buffer1, input_buffer2, pc):
        assert data_memory_size > 0, "Data_memory size should be non-zero"
        self.data_memory_size = data_memory_size
        self.data = data
        self.addres_register = 0
        self.data_operand = 0
        self.program_counter = pc
        self.acc = 0
        self.ALU = ALU()
        self.input_buffer1 = input_buffer1
        self.input_buffer2 = input_buffer2
        self.output_buffer1 = []
        self.output_buffer2 = []

    def signal_input(self):
        # было принято непростое решение сделать первый поток ввода для символов, второй для чисел
        num = self.data_operand
        try:
            if num==1:
                byt, self.input_buffer1 = self.input_buffer1[0], self.input_buffer1[1:]
                self.acc = ord(byt)
                logging.debug("input: %s", repr(chr(self.acc)))
            if num==2:
                num, self.input_buffer2 = self.input_buffer2[0], self.input_buffer2[1:]
                self.acc = num
                logging.debug("input: %s", repr(self.acc))
        except:
            assert 0, "read from empty buffer"

    # def signal_latch_data_operand(self):
    #     self.data_operand = self.data[self.addres_register:self.addres_register+4]
    def signal_latch_addres_reg(self, sel):
        if sel == Selector.PC:
            self.addres_register = self.program_counter
        elif sel == Selector.ALU:
            self.addres_register = self.ALU.out
        elif sel == Selector.ACC:
            self.addres_register = self.acc

        assert 0 <= self.addres_register < self.data_memory_size, "out of memory: {}".format(self.addres_register)

    def signal_latch_acc(self):
        self.acc = self.ALU.out
    def signal_read(self):
        self.data_operand = byte_to_int(self.data[self.addres_register:self.addres_register+4])
    def signal_write(self):
        self.data[self.addres_register:self.addres_register+4] = int_to_bytes(self.acc)


    def signal_output(self):
        num = self.data_operand
        if num == 1:
            logging.debug("output1: %s << %s", repr("".join(map(chr, self.output_buffer1))), chr(self.acc))
            self.output_buffer1.append(self.acc)

        if num == 2:
            logging.debug("output:2 %s << %s", repr(" ".join(map(str, self.output_buffer2))), self.acc)
            self.output_buffer2.append(self.acc)

    def signal_latch_ALU_right_input(self):
        self.ALU.right_input = np.int32(self.acc)
    def signal_latch_ALU_left_input(self):
        self.ALU.left_input = np.int32(self.data_operand)
    def signal_latch_pc(self, sel):
        if sel is Selector.ADD1:
            self.program_counter += 1
        if sel is Selector.ADD4:
            self.program_counter += 4
        if sel is Selector.DATA_OPERAND:
            self.program_counter = self.data_operand

    def zero(self):
        """Флаг нуля. Необходим для условных переходов."""
        return self.acc == 0
    def neg(self):
        """Флаг нуля. Необходим для условных переходов."""
        return self.acc < 0
    def overflow(self):
        """Флаг нуля. Необходим для условных переходов."""
        return self.ALU.V
    def carry(self):
        """Флаг нуля. Необходим для условных переходов."""
        return self.ALU.C

    def get_opcode(self):
        return int_to_bytes(self.data_operand)[0]




class ControlUnit:


    # data = None
    # "Память команд."
    """Структура микрокоманды(бит/бит:бит - назначение) 1 - вызвать сигнал, 0 - не вызывать:
        0 - signal_latch_mpc(0 - +1, 1 - LUT(DO))
        1 - signal_latch_ALU_right_input(DP)
        2 - signal_latch_ALU_left_input(DP)
        3 - signal_add(ALU)
        4 - signal_sub(ALU)
        5 - signal_mul(ALU)                 В общем говоря сигналы АЛУ можно было закодировать 3 битами, т.к. они не могут выполнятся одновременно, но это усложнило бы модель и интерпретатор
        6 - signal_mul_high(ALU)
        7 - signal_not(ALU)
        8 - signal_or(ALU)
        9 - signal_and(ALU)
        10 - signal_input(DP)
        11 - signal_latch_acc(DP)
        12 - signal_output(DP)
        #### 17 - signal_read_micro_command удален за ненадобностью, мы всегда читаем новую микрокоманду
        13:14 - signal_latch_1_pc(00 - don't latch, 01 - right input on schema(+1), 10 - left(зависит от другого мультиплексора))
        15:17 - signal_choose_flag (000 - instant false, 001 - carry_flag, 010 - zero_flag, 011 - neg_flag, 100 - overflow_flag, 101 - instant true)
        18:19 - signal_latch_addres_reg(DP)       (00 - don't latch, 01 - from ALU, 10 = from ACC, 11 - from PС)
        20 - signal_read(DP)
        21 - signal_write(DP)
        22 - halt
    """
    LUT = {
            '0x00': 2,
            '0x01': 5,
            '0x02': 11,
            '0x03': 29,
            '0x04': 35,
            '0x05': 41,
            '0x06': 53,
            '0x07': 59,
            '0x08': 65,
            '0x09': 69,
            '0x0a': 73,
            '0x0b': 77,
            '0x0c': 81,
            '0x0d': 85,
            '0x0e': 91,
            '0x0f': 15,
            '0x10': 23,
            '0x11': 47,
            '0x12': 4
    }
    microprogram_memory = [
        '0 000000000 0 0 0 00 000 11 1 0 0', #pc -> ar; mem[ar] -> data_operand, нужно для начала чтения программы(получения первого опкода)
        '1 000000000 0 0 0 01 000 00 0 0 0', #pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 101000100 0 0 0 00 000 11 1 0 0', #NOT       acc -> left_ALU, ALU add, ALU not, pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 1 0 01 000 00 0 0 0',

        '0 000000000 0 0 0 00 000 00 0 0 1', #HALT

        '0 000000000 0 0 0 00 000 11 1 0 0', #READ arg_addr -> data_op                                         |
        '0 010000010 0 0 0 10 000 01 0 0 0', # data_op+0 -> addr_reg, pc + 4 -> pc                             | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0', # mem[addr_reg] -> data_op                                        |
        '0 011000000 0 1 0 00 000 11 0 0 0', # arg + 0 -> acc,pc -> ar
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0', #pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # WRITE arg_addr -> data_op
        '0 010000010 0 0 0 10 000 01 0 1 0',  # data_op+0 -> addr_reg, acc -> mem[ar]
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand, нужно для начала чтения программы(получения первого опкода)
        '1 000000000 0 0 0 01 000 00 0 0 0',

        '0 000000000 0 0 0 00 000 11 1 0 0',  # READ_IND arg_addr -> data_op          |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 011000000 0 0 0 00 000 01 0 0 0',  #
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '0 011000000 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # WRITE_IND arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 011000000 0 0 0 00 000 01 0 1 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # ADD       arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 111000000 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # SUB       arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 110100000 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # MUL       arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 110010000 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # MUL_HIGH  arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 110001000 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # AND       arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 110000001 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # OR        arg_addr -> data_op         |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 110000010 0 1 0 00 000 11 0 0 0',
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',   # BEQ        arg_addr -> data_op
        '0 000000000 0 0 0 10 010 00 0 0 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # BNE        arg_addr -> data_op
        '0 000000000 0 0 0 10 011 00 0 0 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # BVS        arg_addr -> data_op
        '0 000000000 0 0 0 10 100 00 0 0 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # BVC        arg_addr -> data_op
        '0 000000000 0 0 0 10 001 00 0 0 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # JUMP        arg_addr -> data_op
        '0 000000000 0 0 0 10 101 00 0 0 0',
        '0 000000000 0 0 0 00 000 11 1 0 0',  # pc -> ar; mem[ar] -> data_operand
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # INPUT        arg_addr -> data_op      |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 000000000 1 0 0 00 000 11 0 0 0',  # input -> acc, pc -> ar; mem[ar] -> data_operand
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен

        '0 000000000 0 0 0 00 000 11 1 0 0',  # INPUT        arg_addr -> data_op      |
        '0 010000010 0 0 0 10 000 01 0 0 0',  # data_op+0 -> addr_reg, pc + 4 -> pc   | - read_addr()
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op              |
        '0 000000000 0 0 1 00 000 11 0 0 0',  # input -> acc, pc -> ar; mem[ar] -> data_operand
        '0 000000000 0 0 0 00 000 00 1 0 0',  # mem[addr_reg] -> data_op
        '1 000000000 0 0 0 01 000 00 0 0 0',  # pc+1 -> pc; адрес микропрограммы, соотвтествующей следующему опкоду загружен
    ]
    data_path = None
    mpc = None
    current_micro_command = None


    _tick = None
    "Текущее модельное время процессора (в тактах). Инициализируется нулём."

    def __init__(self, data_path):
        # self.data = data
        self.data_path = data_path
        self.mpc = 0
        self._tick = 0

    def parse_microcomand(self, microcomand):
        microcomand = microcomand.replace(" ", "")
        if microcomand[0] == '0':
            self.signal_latch_mpc(Selector.ADD1)
        else:
            self.signal_latch_mpc(Selector.LUT)
        if microcomand[1] == '1':
            self.data_path.signal_latch_ALU_right_input()

        if microcomand[2] == '1':
            self.data_path.signal_latch_ALU_left_input()
        if microcomand[3] == '1':
            self.data_path.ALU.signal_add()
        if microcomand[4] == '1':
            self.data_path.ALU.signal_sub()
        if microcomand[5] == '1':
            self.data_path.ALU.signal_mul()
        if microcomand[6] == '1':
            self.data_path.ALU.signal_mul_high()
        if microcomand[7] == '1':
            self.data_path.ALU.signal_not()
        if microcomand[8] == '1':
            self.data_path.ALU.signal_or()
        if microcomand[9] == '1':
            self.data_path.ALU.signal_and()
        if microcomand[10] == '1':
            self.data_path.signal_input()
        if microcomand[11] == '1':
            self.data_path.signal_latch_acc()
        if microcomand[12] == '1':
            self.data_path.signal_output()
        if microcomand[13:15] == '00':
            pass
        elif microcomand[13:15] == '01':
            self.data_path.signal_latch_pc(Selector.ADD1)
        else:
            if microcomand[15:18] == '000':
                self.data_path.signal_latch_pc(Selector.ADD4)
            if microcomand[15:18] == '001':
                if self.data_path.carry():
                    self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
                else:
                    self.data_path.signal_latch_pc(Selector.ADD4)

            if microcomand[15:18] == '010':
                if self.data_path.zero():
                    self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
                else:
                    self.data_path.signal_latch_pc(Selector.ADD4)

            if microcomand[15:18] == '011':
                if self.data_path.neg():
                    self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
                else:
                    self.data_path.signal_latch_pc(Selector.ADD4)

            if microcomand[15:18] == '100':
                if self.data_path.overflow():
                    self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
                else:
                    self.data_path.signal_latch_pc(Selector.ADD4)
            if microcomand[15:18] == '101':
                self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
        if microcomand[18:20] == '00':
            pass
        elif microcomand[18:20] == '01':

            self.data_path.signal_latch_addres_reg(Selector.ALU)
        elif microcomand[18:20] == '10':
            self.data_path.signal_latch_addres_reg(Selector.ACC)
        else:
            self.data_path.signal_latch_addres_reg(Selector.PC)
        if microcomand[20] == '1':
            self.data_path.signal_read()
        if microcomand[21] == '1':
            self.data_path.signal_write()
        if microcomand[22] == '1':
            raise StopIteration()
    def tick(self):
        """Продвинуть модельное время процессора вперёд на один такт."""
        self._tick += 1

    def current_tick(self):
        """Текущее модельное время процессора (в тактах)."""
        return self._tick
    def signal_latch_mpc(self, sel):
        if sel is Selector.ADD1:
            self.mpc += 1
        if sel is Selector.LUT:
            self.mpc = self.LUT[self.data_path.get_opcode()]

    def read_micro_command(self):
        self.current_micro_command = self.microprogram_memory[self.mpc]
    def read_arg(self):
        #читает аргумент в data_operand
        self.data_path.signal_latch_addres_reg(Selector.PC)
        self.data_path.signal_read()

        self.data_path.signal_latch_ALU_left_input()
        self.data_path.ALU.signal_or()
        self.data_path.signal_latch_addres_reg(Selector.ALU)
        self.data_path.signal_read()
        self.data_path.signal_latch_pc(Selector.ADD4)

    def process_next_micro_command(self):  # noqa: C901 # код имеет хорошо структурирован, по этому не проблема.
        """Основной цикл процессора. Декодирует и выполняет инструкцию.

        Обработка инструкции:

        1. Проверить `Opcode`.

        2. Вызвать методы, имитирующие необходимые управляющие сигналы.

        3. Продвинуть модельное время вперёд на один такт (`tick`).

        4. (если необходимо) повторить шаги 2-3.

        5. Перейти к следующей инструкции.

        Обработка функций управления потоком исполнения вынесена в
        `decode_and_execute_control_flow_instruction`.
        """
        self.read_micro_command()
        self.parse_microcomand(self.current_micro_command)
        self.tick()

    def __repr__(self):
        """Вернуть строковое представление состояния процессора."""
        state_repr = "TICK: {:3} MPC: {} PC: {:3} ADDR: {:3} DATA_OP: {} ACC: {}".format(
            self._tick,
            self.mpc,
            self.data_path.program_counter,
            self.data_path.addres_register,
            self.data_path.data_operand,
            self.data_path.acc,
        )

        # instr = self.program[self.program_counter]
        # opcode = instr["opcode"]
        # instr_repr = str(opcode)
        #
        # if "arg" in instr:
        #     instr_repr += " {}".format(instr["arg"])
        #
        # instr_hex = f"{opcode_to_binary[opcode] << 28 | (instr.get('arg', 0) & 0x0FFFFFFF):08X}"
        #
        # return "{} \t{} [{}]".format(state_repr, instr_repr, instr_hex)
        return state_repr


def simulation(data, input_1, input_2, data_memory_size, limit, pc):
    """Подготовка модели и запуск симуляции процессора.

    Длительность моделирования ограничена:

    - количеством выполненных тактов (`limit`);

    - количеством данных ввода (`input_tokens`, если ввод используется), через
      исключение `EOFError`;

    - инструкцией `Halt`, через исключение `StopIteration`.
    """
    data_path = DataPath(data_memory_size,data, input_1, input_2, pc)
    control_unit = ControlUnit(data_path)

    logging.debug("%s", control_unit)
    try:
        while control_unit._tick < limit:
            control_unit.process_next_micro_command()
            logging.debug("%s", control_unit)
    except EOFError:
        logging.warning("Input buffer is empty!")
    except StopIteration:
        pass

    if control_unit._tick >= limit:
        logging.warning("Limit exceeded!")
    logging.info("output_buffer1: %s", repr("".join(map(chr, data_path.output_buffer1))))
    logging.info("output_buffer2: %s", repr(" ".join(map(str, data_path.output_buffer2))))
    logging.info("general tick count: %s", repr(control_unit.current_tick()))

    return " ".join(map(str, data_path.output_buffer2)), " ".join(map(str, data_path.output_buffer1)), control_unit.current_tick()


def main(code_file, input_file, quiet_flag):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    """Функция запуска модели процессора. Параметры -- имена файлов с машинным
    кодом и с входными данными для симуляции.
    """

    #так как для удобства ввода формирование паскаль строки и длины массива происходит вне входного файла(здесь) нам нужно различать ввод символов и чисел

    if quiet_flag:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    with open(code_file, "rb") as file:

        binary_code = file.read()
    data = [f"0x{byte:02x}" for byte in binary_code]
    start_ind, data = byte_to_int(data[:4]), data[4:]


    symbol_file = True
    input_1 = []
    input_2 = []
    #надо решить че там с двумя потоками ввода
    with open(input_file, encoding="utf-8") as file:
        input_text = file.read()
        input_token = []
        if input_text[0]=='n':
            line = file.readline()
            mass = []
            for i in input_text.split()[1:]:
                mass = mass + [int(i)]
            input_2 = [len(mass)] + mass
            print(input_2)
        else:
            for char in input_text[1:]:
                input_token.append(char)
            input_1 = list(int_to_chars(len(input_token))) + input_token
    output1, output2, ticks = simulation(
        data,
        input_1=input_1,
        input_2=input_2,
        data_memory_size=15000,
        limit=50000,
        pc=start_ind
    )


original_stdout = sys.stdout
original_stderr = sys.stderr
if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if quiet:
        sys.argv.remove("--quiet")
        assert len(sys.argv) == 3, "Usage: machine.py <code_file> <input_file>"
        _, code_file, input_file = sys.argv
        main(code_file, input_file, quiet)
    else:
        assert len(sys.argv) == 3, "Usage: machine.py <code_file> <input_file>"
        _, code_file, input_file = sys.argv
        main(code_file, input_file, quiet)
