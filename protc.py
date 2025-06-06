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

import numpy as np

from isa import Opcode, from_bytes, opcode_to_binary, binary_to_opcode
from translator import byte_to_int, int_to_bytes, int_to_chars


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
        print(x, y)
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

    def signal_input(self, num):
        print(self.input_buffer2)
        # было принято непростое решение сделать первый поток ввода для символов, второй для чисел
        try:
            if num==1:
                byt, self.input_buffer1 = self.input_buffer1[0], self.input_buffer1[1:]
                self.acc = ord(byt)
            if num==2:
                num, self.input_buffer2 = self.input_buffer2[0], self.input_buffer2[1:]
                self.acc = num
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
        """Защёлкнуть слово из памяти (`oe` от Output Enable) и защёлкнуть его в
        аккумулятор. Сигнал `oe` выставляется неявно `ControlUnit`-ом.
        """
        self.acc = self.ALU.out
    def signal_read(self):
        self.data_operand = byte_to_int(self.data[self.addres_register:self.addres_register+4])
    def signal_write(self):
        self.data[self.addres_register:self.addres_register+4] = int_to_bytes(self.acc)
        # logging.debug("input: %s", repr(symbol))

    def signal_output(self, num):
        # logging.debug("output: %s << %s", repr("".join(self.output_buffer)), repr(symbol));

        if num == 1:
            self.output_buffer1.append(self.acc)
        if num == 2:
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



# microcode = [
#
# ]
#
#
#
# microcode_interpretator=[
#     self.signal_add
# ]
# def parse_microcode(microcode):



class ControlUnit:
    """Блок управления процессора. Выполняет декодирование инструкций и
    управляет состоянием модели процессора, включая обработку данных (DataPath).

    Согласно варианту, любая инструкция может быть закодирована в одно слово.
    Следовательно, индекс памяти команд эквивалентен номеру инструкции.

    ```text
    +------------------(+1)-------+
    |                             |
    |   +-----+                   |
    +-->|     |     +---------+   |    +---------+
        | MUX |---->| program |---+--->| program |
    +-->|     |     | counter |        | memory  |
    |   +-----+     +---------+        +---------+
    |      ^                               |
    |      | sel_next                      | current instruction
    |      |                               |
    +---------------(select-arg)-----------+
           |                               |      +---------+
           |                               |      |  step   |
           |                               |  +---| counter |
           |                               |  |   +---------+
           |                               v  v        ^
           |                       +-------------+     |
           +-----------------------| instruction |-----+
                                   |   decoder   |
                                   |             |<-------+
                                   +-------------+        |
                                           |              |
                                           | signals      |
                                           v              |
                                     +----------+  zero   |
                                     |          |---------+
                                     | DataPath |
                      input -------->|          |----------> output
                                     +----------+
    ```

    """

    # data = None
    # "Память команд."

    data_path = None
    "Блок обработки данных."

    _tick = None
    "Текущее модельное время процессора (в тактах). Инициализируется нулём."

    def __init__(self, data, data_path):
        # self.data = data
        self.data_path = data_path
        self._tick = 0
        self.step = 0

    def tick(self):
        """Продвинуть модельное время процессора вперёд на один такт."""
        self._tick += 1

    def current_tick(self):
        """Текущее модельное время процессора (в тактах)."""
        return self._tick

    # def signal_latch_program_counter(self, addr):
    #     """Защёлкнуть новое значение счётчика команд.
    #
    #     Если `sel_next` равен `True`, то счётчик будет увеличен на единицу,
    #     иначе -- будет установлен в значение аргумента текущей инструкции.
    #     """
    #     if not addr:
    #         self.program_counter += 1
    #     else:
    #         self.program_counter = addr
    def read_arg(self):
        #читает аргумент в data_operand
        self.data_path.signal_latch_addres_reg(Selector.PC)
        self.data_path.signal_read()

        self.data_path.signal_latch_ALU_left_input()
        self.data_path.ALU.signal_or()
        self.data_path.signal_latch_addres_reg(Selector.ALU)
        self.data_path.signal_read()
        self.data_path.signal_latch_pc(Selector.ADD4)

    def process_next_command(self):  # noqa: C901 # код имеет хорошо структурирован, по этому не проблема.
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
        self.data_path.signal_latch_addres_reg(Selector.PC)
        self.data_path.signal_read()
        opcode = binary_to_opcode[self.data_path.get_opcode()]
        print(opcode)

        self.data_path.signal_latch_pc(Selector.ADD1)
        if opcode is Opcode.HALT:
            print(self.data_path.data[:100])
            raise StopIteration()

        if opcode is Opcode.NOT:
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_add()
            self.data_path.ALU.signal_not()
            self.data_path.signal_latch_acc()
            return

        if opcode is Opcode.READ:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.ALU.signal_add()

            self.data_path.signal_latch_acc()
            return

        if opcode is Opcode.WRITE:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()

            self.data_path.signal_latch_ALU_left_input()
            self.data_path.ALU.signal_add()
            self.data_path.signal_latch_addres_reg(Selector.ALU)

            self.data_path.signal_write()
            self.data_path.signal_latch_pc(Selector.ADD4)
            return

        if opcode is Opcode.READ_IND:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.ALU.signal_add()
            self.data_path.signal_latch_addres_reg(Selector.ALU)
            self.data_path.signal_read()

            self.data_path.signal_latch_ALU_left_input()
            self.data_path.ALU.signal_add()
            self.data_path.signal_latch_acc()

        if opcode is Opcode.WRITE_IND:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.ALU.signal_add()
            self.data_path.signal_latch_addres_reg(Selector.ALU)

            self.data_path.signal_write()

        if opcode is Opcode.ADD:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_add()

            self.data_path.signal_latch_acc()
        if opcode is Opcode.SUB:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_sub()

            self.data_path.signal_latch_acc()
        if opcode is Opcode.MUL:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_mul()
            print(self.data_path.ALU.out)

            self.data_path.signal_latch_acc()
        if opcode is Opcode.AND:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_and()

            self.data_path.signal_latch_acc()
        if opcode is Opcode.OR:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_or()

            self.data_path.signal_latch_acc()

        if opcode is Opcode.BNE:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()

            if self.data_path.neg():
                self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
            else:
                self.data_path.signal_latch_pc(Selector.ADD4)

        if opcode is Opcode.BEQ:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()

            if self.data_path.zero():
                self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
            else:
                self.data_path.signal_latch_pc(Selector.ADD4)
        if opcode is Opcode.BVS:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()

            if self.data_path.overflow():
                self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
            else:
                self.data_path.signal_latch_pc(Selector.ADD4)
        if opcode is Opcode.BCS:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()

            if self.data_path.carry():
                self.data_path.signal_latch_pc(Selector.DATA_OPERAND)
            else:
                self.data_path.signal_latch_pc(Selector.ADD4)

        if opcode is Opcode.JUMP:
            self.data_path.signal_latch_addres_reg(Selector.PC)
            self.data_path.signal_read()
            self.data_path.signal_latch_pc(Selector.DATA_OPERAND)

        if opcode is Opcode.INPUT:
            self.read_arg()
            self.data_path.signal_input(self.data_path.data_operand)

        if opcode is Opcode.OUTPUT:
            self.read_arg()
            print("debug: ", self.data_path.data_operand)
            self.data_path.signal_output(self.data_path.data_operand)
        if opcode is Opcode.MUL_HIGH:
            self.read_arg()
            self.data_path.signal_latch_ALU_left_input()
            self.data_path.signal_latch_ALU_right_input()
            self.data_path.ALU.signal_mul_high()
            print(self.data_path.ALU.out)

            self.data_path.signal_latch_acc()
        # if opcode in {Opcode.RIGHT, Opcode.LEFT}:
        #     self.data_path.signal_latch_data_addr(opcode.value)
        #     self.signal_latch_program_counter(sel_next=True)
        #     self.step = 0
        #     self.tick()
        #     return
        #
        # if opcode in {Opcode.INC, Opcode.DEC, Opcode.INPUT}:
        #     if self.step == 0:
        #         self.data_path.signal_latch_acc()
        #         self.step = 1
        #         self.tick()
        #         return
        #     if self.step == 1:
        #         self.data_path.signal_wr(opcode.value)
        #         self.signal_latch_program_counter(sel_next=True)
        #         self.step = 0
        #         self.tick()
        #         return
        #
        # if opcode is Opcode.PRINT:
        #     if self.step == 0:
        #         self.data_path.signal_latch_acc()
        #         self.step = 1
        #         self.tick()
        #         return
        #     if self.step == 1:
        #         self.data_path.signal_output()
        #         self.signal_latch_program_counter(sel_next=True)
        #         self.step = 0
        #         self.tick()
        #         return

    def __repr__(self):
        """Вернуть строковое представление состояния процессора."""
        state_repr = "TICK: {:3} PC: {:3}/{} ADDR: {:3} DATA_OP: {} ACC: {}".format(
            self._tick,
            self.data_path.program_counter,
            self.step,
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
    control_unit = ControlUnit(data, data_path)

    logging.debug("%s", control_unit)
    try:
        while control_unit._tick < limit:
            control_unit.process_next_command()
            logging.debug("%s", control_unit)
    except EOFError:
        logging.warning("Input buffer is empty!")
    except StopIteration:
        pass

    if control_unit._tick >= limit:
        logging.warning("Limit exceeded!")
    logging.info("output_buffer1: %s", repr(" ".join(map(chr, data_path.output_buffer1))))
    logging.info("output_buffer2: %s", repr(" ".join(map(str, data_path.output_buffer2))))

    return " ".join(map(str, data_path.output_buffer2)), " ".join(map(str, data_path.output_buffer1)), control_unit.current_tick()


def main(code_file, input_file):
    """Функция запуска модели процессора. Параметры -- имена файлов с машинным
    кодом и с входными данными для симуляции.
    """

    #так как для удобства ввода формирование паскаль строки и длины массива происходит вне входного файла(здесь) нам нужно различать ввод символов и чисел

    with open(code_file, "rb") as file:

        binary_code = file.read()
    data = [f"0x{byte:02x}" for byte in binary_code]
    start_ind, data = byte_to_int(data[:4]), data[4:]


    symbol_file = True
    input_1 = []
    input_2 = []
    #надо решить че там с двумя потоками ввода
    with open(input_file, encoding="utf-8") as file:
        if ".num" in input_file:
            symbol_file = False
        if symbol_file:
            input_text = file.read()
            input_token = []
            for char in input_text:
                input_token.append(char)
            input_1 = list(int_to_chars(len(input_token))) + input_token
        else:
            line = file.readline()
            mass = []
            for i in line.split():
                mass = mass + [int(i)]
            input_2 = [len(mass)] + mass

    output1, output2, ticks = simulation(
        data,
        input_1=input_1,
        input_2=input_2,
        data_memory_size=15000,
        limit=2000,
        pc=start_ind
    )


    print("".join(output1))
    print("".join(output2))
    print("ticks:", ticks)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Wrong arguments: machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)