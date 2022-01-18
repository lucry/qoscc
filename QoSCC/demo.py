import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')

    config = parser.parse_args()
    print('--eval', config.eval)
    print('--load', config.load)