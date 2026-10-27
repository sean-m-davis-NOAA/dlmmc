#!/usr/bin/env python3
"""Precompile the Stan models shipped with the dlmmc package."""

import argparse

import dlmmc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'models',
        nargs='*',
        help='model names to compile (default: all packaged models)',
    )
    args = parser.parse_args()
    for model_name, executable in dlmmc.compile_models(args.models or None).items():
        print(f'{model_name}: {executable}')


if __name__ == '__main__':
    main()
