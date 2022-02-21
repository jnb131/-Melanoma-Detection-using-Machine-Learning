import os
import argparse
from shutil import copy
from pathlib import Path

# go through entire input directory with train_sep, valid, and test
# for each subdir melanoma and not melanoma, downselect to 500 or 100 examples each (1000/2 or 200/2)
# output to output dir

def dir_existing(dir):
    print(dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def downselect_examples_category(src_dir, dest_dir, num_examples):
    examples = [filename for filename in os.listdir(src_dir) if not filename.startswith('.')]

    for i in range(num_examples):
        example_name = examples[i]
        copy(src_dir / example_name, dest_dir / example_name)

def downselect_examples_set(src_dir, dest_dir, num_examples):
    classes = [filename for filename in os.listdir(src_dir) if not filename.startswith('.')]

    for c in classes:
        downselect_examples_category(src_dir / c, dir_existing(dest_dir / c), num_examples // 2)

def downselect_examples_macro_dataset(src_dir, dest_dir, num_examples):
    sets = [filename for filename in os.listdir(src_dir) if not filename.startswith('.')]
    dir_existing(dest_dir)

    for s in sets:
        src_set = src_dir / s
        dest_set = dest_dir / s
        
        downselect_examples_set(src_set, dir_existing(dest_set), num_examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downselect dataset to size')
    parser.add_argument('source_dir')
    parser.add_argument('destination_dir')
    parser.add_argument('size', type=int)
    args = parser.parse_args()

    downselect_examples_macro_dataset(Path(args.source_dir), Path(args.destination_dir), args.size)
    
