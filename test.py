import numpy as np
np.seterr(over='ignore')
from translator import int_to_bytes

# n = np.int32(22)
# print(n+1)
# try:
#     print(n+2**31)
# except:
#     print("c")
#     print(-2**31+2**30+n*1000000)
# print(~n+1)
x = np.int32(-2**31)
y = np.int32(2**31-1)
print(-
      x)
print(x)


def signal_add(x, y):
    out = x + y
    sum_uint = x.astype(np.uint32) + y.astype(np.uint32)
    C = sum_uint < x.astype(np.uint32)
    V = ((x ^ y) >= 0) & ((x ^ out) < 0)

    print("Результат:", out)
    print("Переполнение (V):", V)
    print("Перенос (C):", C)
def signal_mul(x, y):
    out = x * y
    sum_int = x.astype(int) * y.astype(int)
    V = sum_int > out
    print(out, V)
x = np.int32(2**17+2)
y = np.int32(2**16)
signal_mul(x, y)
# print(int_to_bytes(x))
#
# def not_32bit(x):
#     return (~x) & 0xFFFFFFFF
# print(not_32bit(1))