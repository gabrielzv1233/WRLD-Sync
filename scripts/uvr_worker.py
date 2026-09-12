"""Isolated UVR inference; the parent owns cancellation and cache publication."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    from audio_separator.separator import Separator
    separator = Separator(
        model_file_dir=args.model_dir, output_dir=args.output_dir,
        output_format='FLAC', output_single_stem='Vocals',
    )
    separator.load_model(model_filename=args.model)
    outputs = separator.separate(args.source)
    directory = Path(args.output_dir)
    candidates = [Path(name) for name in (outputs or [])]
    candidates = [p if p.is_absolute() else directory / p for p in candidates]
    vocal = next((p for p in candidates if p.is_file() and 'vocal' in p.name.lower()), None)
    if vocal is None:
        raise RuntimeError('UVR separation completed without a vocals stem')
    # Publish only the path, after separation and encoding have both finished.
    (directory / 'result.txt').write_text(str(vocal.resolve()), encoding='utf-8')


if __name__ == '__main__':
    main()
