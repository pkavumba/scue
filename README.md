# scue

## Usage

`scue` supports

- Python 3.7 and up (CPython and PyPy)
- Rust 1.56 and up

```python
import argparse

from scue.models import (
    UnbalancedNgramModelForMultipleChoice,
    SequenceLengthModelForMultipleChoice,
    LexicalOverlapModelForMultipleChoice,
    EnsembleModelForMultipleChoice,
)
from scue.preprocessors import MultipleChoicePreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=False)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--id_field", type=str, required=True)
    parser.add_argument("--context_field", type=str, required=True)
    parser.add_argument("--choices_field", type=str, required=True)
    parser.add_argument("--question_field", type=str, required=False)
    args = parser.parse_args()

    processor = MultipleChoicePreprocessor(
        id_field=args.id_field,
        context_field=args.context_field,
        choices_field=args.choices_field,
        remove_stopwords=False,
        remove_punctuation=False,
    )
    data = processor(args.train_file)

    model = UnbalancedNgramModelForMultipleChoice(n=args.n, training_data=data)
    model.fit()

    # accuracy on training set
    print(model.evaluate(data))

    # print important features
    print(model.important_features)

    if args.validation_file:
        val_data = processor(args.validation_file)
        print(model.evaluate(val_data))


```

## Development

`scue` requires

- Python 3.7 and up (CPython and PyPy).
- Rust 1.56 and up

First, follow the commands below to create a new directory containing a new Python virtualenv, and install `maturin` into the virtualenv using Python's package manager, pip:

```bash
$ git clone
$ cd scue
$ mkdir string_sum
$ cd string_sum
$ python -m venv .env
$ source .env/bin/activate
$ pip install maturin
```

Finally, run `maturin` develop. This will build the rust buckend package and install it into the Python virtualenv previously created and activated. The package is then ready to be used from `python`:

```bash
$ maturin develop
```

## License

`scue` is licensed under the Apache-2.0 license.
