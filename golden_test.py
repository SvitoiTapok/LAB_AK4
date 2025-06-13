import pytest
import contextlib
import io
import logging
import os
import tempfile
import machine
import translator
MAX_LOG = 400000000




@pytest.mark.golden_test("golden/hello_world.yml")
def test_translator_and_machine_hello_world(golden, caplog):
    run_test(golden, caplog, False)

@pytest.mark.golden_test("golden/cat.yml")
def test_translator_and_machine_cat(golden, caplog):
    run_test(golden, caplog, False)
@pytest.mark.golden_test("golden/hello_user_name_1.yml")
def test_translator_and_machine_hello_user_name_1(golden, caplog):
    run_test(golden, caplog, False)

@pytest.mark.golden_test("golden/hello_user_name_2.yml")
def test_translator_and_machine_hello_user_name_2(golden, caplog):
    run_test(golden, caplog, False)
#

#
@pytest.mark.golden_test("golden/new_sort.yml")
def test_translator_and_machine_new_sort(golden, caplog):
    run_test(golden, caplog, False)

@pytest.mark.golden_test("golden/arith_64_add.yml")
def test_translator_and_machine_carry_check(golden, caplog):
    run_test(golden, caplog, False)

@pytest.mark.golden_test("golden/arith_64_mul.yml")
def test_translator_and_machine_mul_high(golden, caplog):
    run_test(golden, caplog, False)

@pytest.mark.golden_test("golden/euler_task.yml")
def test_translator_and_machine_prob1(golden, caplog):
    run_test(golden, caplog, False)

def run_test(golden, caplog, quiet_flag):
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdirname:
        source = os.path.join(tmpdirname, "in_code")
        input_stream = os.path.join(tmpdirname, "input_stream")
        target = os.path.join(tmpdirname, "target.bin")
        target_hex = os.path.join(tmpdirname, "target.bin.hex")

        with open(source, "w", encoding="utf-8") as file:
            file.write(golden["in_code"])
        with open(input_stream, "w", encoding="utf-8") as file:
            file.write(golden["in_stdin"])

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            translator.main(source, target)
            print("============================================================")
            machine.main(target, input_stream, quiet_flag)

        with open(target_hex, encoding="utf-8") as file:
            code_hex = file.read()
        if not quiet_flag:
            assert code_hex == golden.out["out_code_hex"]
            assert stdout.getvalue() == golden.out["out_stdout"]
            assert caplog.text[:MAX_LOG] + "EOF" == golden.out["out_log"]