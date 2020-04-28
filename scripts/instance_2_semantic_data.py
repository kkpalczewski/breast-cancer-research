import argparse


def main():
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--data_config_path', type=str, help='path to data generation config')
    args = parser.parse_args()


if __name__ == "__main__":
    main()
