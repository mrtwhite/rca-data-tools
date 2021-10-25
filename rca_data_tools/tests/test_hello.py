from rca_data_tools import hello


def test_hello():
    hello_text = hello.say_hello()

    assert hello_text == "Hello world."
