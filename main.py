import argparse

from train import start

parser = argparse.ArgumentParser()

parser.add_argument(
    "-j", "--joystick", default=False, action="store_true", help="On/Off joystick mode")


if __name__ == '__main__':
    args = parser.parse_args()
    start(args)