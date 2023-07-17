# scue

## Usage

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
