## Note:
* `nntools.py` is Neural Network tools developed for UCSD ECE285 MLIP. Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
* 'nntools.py, inference.py, main.py, args.py' modified by Kibernetika Inc. for the purpose of quantization experiments


## Kibernetika Quantization Experiments

## Model Training without Quantization
In Kibernetika run no-qat task

python $SRC_DIR /main.py --root_dir $DATA_DIR/images --output_dir $TRAINING_DIR/no_qat --num_epochs 20  --model=dudncnn

## Quantization Aware Model Training
In Kibernetika.AI run qat task

python $SRC_DIR/main.py --root_dir $DATA_DIR/images --output_dir $TRAINING_DIR/qat --num_epochs 20 --model quantize

## Quantization Aware Model Training with fx_static quantization
In Kibernetika.AI run qat_fx_static

python $SRC_DIR/main.py --root_dir $DATA_DIR/images --output_dir $TRAINING_DIR/qat_fx_static --num_epochs 1 --model quantize --quantize fx_static
